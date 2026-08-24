"""A collection of functional utilities for OCL."""

from torch import Tensor
from torch.nn.functional import kl_div, log_softmax


def hinton_distillation_loss(
    teacher_logits: Tensor, student_logits: Tensor, temperature: float = 1.0
) -> Tensor:
    """Hinton's distillation loss [#f1]_ .

    .. math::
        L_{KD} = T^2 KL(softmax(z_s / T), softmax(z_t / T))

    where :math:`T` is the temperature, :math:`z_s` are the student logits, and
    :math:`z_t` are the teacher logits.

    ..  [#f1] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a
        Neural Network. arXiv:1503.02531 [Cs, Stat]. http://arxiv.org/abs/1503.02531

    :param teacher_logits: Teacher logits of shape ``(batch_size, num_classes)``.
    :param student_logits: Student logits of shape ``(batch_size, num_classes)``.
    :param temperature: Temperature for distillation. Higher values produce softer
        probability distributions.
    :return: The distillation loss as a scalar tensor.
    """
    return (
        kl_div(
            log_softmax(student_logits / temperature, dim=1),  # Soft predictions
            log_softmax(teacher_logits / temperature, dim=1),  # Soft targets
            log_target=True,
            reduction="batchmean",  # Mathematically correct unlike the default
        )
        * temperature**2
    )
