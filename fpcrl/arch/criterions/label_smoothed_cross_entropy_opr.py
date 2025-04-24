import math
from dataclasses import dataclass, field

import torch
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    use_fus: bool = field(
        default=False,
        metadata={"help:": "use fus loss"},
    )
    fus_warmup: float = field(
        default=1,
        metadata={"help:": "fus loss warmup"},
    )
    use_cst: bool = field(
        default=False,
        metadata={"help:": "use cst loss"},
    )
    cst_warmup: float = field(
        default=1,
        metadata={"help:": "cst loss warmup"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def get_sequence_representation(hidden_states, padding_mask):
    # padding_mask = (~padding_mask).float()
    padding_mask = (~padding_mask)
    seq_hidden = (hidden_states.transpose(0, 1).contiguous() * padding_mask.unsqueeze(-1)).sum(dim=1)
    seq_hidden = seq_hidden / padding_mask.sum(dim=1).unsqueeze(-1)
    return seq_hidden


def _compute_fus_loss(x_irr, x_irr_fus, padding_mask, reduce=True):
    x_irr = get_sequence_representation(x_irr, padding_mask)
    x_irr_fus = get_sequence_representation(x_irr_fus, padding_mask)

    bsz, hsize = x_irr.size()

    # Normalized features
    c_f = F.normalize(x_irr, 2, dim=-1)
    a_f = F.normalize(x_irr_fus, 2, dim=-1)
    temperature = 0.1  # 设置平滑温度

    # Computing contrast loss
    logits = torch.inner(c_f, a_f).float() / temperature  # 计算内积（即相似度）

    # Calculate the similarity probability between two sample pairs, the value range is [0, 1]
    probs = torch.sigmoid(logits)

    # Boolean matrix with True values on the diagonal
    # The probability of screening out positive sample pairs (i.e. the corresponding elements between x_con and x_con_aug)
    p_mask = torch.eye(bsz, bsz, device=probs.device).bool()

    # Calculate the probability of positive and negative samples
    p_probs = probs[p_mask]
    n_probs = (1 - probs[~p_mask]).view(bsz, bsz - 1)

    # Calculating positive and negative losses
    p_loss = p_probs.log()
    n_loss = n_probs.log().sum(-1)

    loss = -(p_loss + n_loss)

    if reduce:
        loss = loss.sum()
    return loss


def _compute_cst_loss(x_ful_pur, y_ful_pur, padding_mask, y_padding_mask):
    x_ful_pur_mean = get_sequence_representation(x_ful_pur, padding_mask)
    y_ful_pur_mean = get_sequence_representation(y_ful_pur, y_padding_mask)
    cst_loss = torch.norm(x_ful_pur_mean - y_ful_pur_mean, p=2)
    return cst_loss


@register_criterion(
    "label_smoothed_cross_entropy_opr", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            use_fus=False,
            fus_warmup=1,
            use_cst=False,
            cst_warmup=1,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.use_fus = use_fus
        self.fus_warmup = fus_warmup
        self.use_cst = use_cst
        self.cst_warmup = cst_warmup

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, fus_loss, cst_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        st_size, fus_size, cst_size = 0, 0, 0

        net_output = model(**sample["net_input"])
        if self.training:
            st_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
            loss = st_loss
            if self.use_fus:
                fus_weight = min((model.num_updates / self.fus_warmup), 1)
                if fus_weight > 0:
                    fus_loss = self.compute_fus_loss(net_output)
                loss = loss + fus_loss * fus_weight
            if self.use_cst:
                cst_weight = min((model.num_updates / self.cst_warmup), 1)
                if cst_weight > 0:
                    cst_loss = self.compute_cst_loss(net_output)
                loss = loss + cst_loss * cst_weight
            st_size = fus_size = cst_size = sample_size = sample["ntokens"]
        else:
            st_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
            loss = st_loss
            st_size = sample_size = sample["ntokens"]

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "fus_loss": fus_loss.data,
            "cst_loss": cst_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "st_sample_size": st_size,
            "fus_sample_size": fus_size,
            "cst_sample_size": cst_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_fus_loss(self, net_output):
        x_irr = net_output[1]["encoder_out"]["encoder_fus_out"][0]
        x_irr_fus = net_output[1]["encoder_out"]["encoder_fus_out"][1]
        padding_mask = net_output[1]["encoder_out"]["encoder_padding_mask"][0]
        fus_loss = _compute_fus_loss(x_irr, x_irr_fus, padding_mask)
        return fus_loss

    def compute_cst_loss(self, net_output):
        x_ful_pur = net_output[1]["encoder_out"]["encoder_cst_out"][0]
        y_ful_pur = net_output[1]["encoder_out"]["encoder_cst_out"][-1]
        padding_mask = net_output[1]["encoder_out"]["encoder_padding_mask"][0]
        y_padding_mask = net_output[1]["encoder_out"]["encoder_padding_mask"][-1]
        cst_loss = _compute_cst_loss(x_ful_pur, y_ful_pur, padding_mask, y_padding_mask)
        return cst_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        fus_loss_sum = sum(log.get("fus_loss", 0) for log in logging_outputs)
        cst_loss_sum = sum(log.get("cst_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        fus_sample_size = sum(log.get("fus_sample_size", 0) for log in logging_outputs)
        cst_sample_size = sum(log.get("cst_sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "fus_loss", fus_loss_sum / fus_sample_size / math.log(2) if fus_sample_size != 0 else 0, fus_sample_size, round=3
        )
        metrics.log_scalar(
            "cst_loss", cst_loss_sum / cst_sample_size / math.log(2) if cst_sample_size != 0 else 0, cst_sample_size, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
