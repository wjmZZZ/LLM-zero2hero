# ================================================================
# Use AI to evaluate the quality of the answer, prompt template
# ================================================================
AI_template_default = """[System]
请对比同一问题下，模型回答和参考答案的优劣。

[问题]
{prompt}

[参考答案开始]
{target_text}
[参考答案结束]

[任务]
现在请您对下面显示的模型的回答进行评价，评分标准包括回答的有用性、相关性、准确性和详细程度。模型的总体评分范围为0到1分，分数越高表示整体表现越好。
0分表示模型无法回答问题，0.5分表示模型部分回答了问题，1分表示完美回答了问题。

请首先提供一份详尽的评价说明。
在最后一行，输出一个单一的分数来表示对模型的评分。
请以结构化的方式分两行给出结果。
explanation: ...
score: ...

[模型回答开始]
{predicted_text}
[模型回答结束]"""

AI_custom_template = ""

import logging
# ================================================================
# function
# ================================================================
from typing import Tuple

from openai import OpenAI

from Others.exceptions import MetricException

logger = logging.getLogger(__name__)


def get_ai_template(template_name: str) -> str:
    """
    Return the corresponding AI evaluation template based on the template name.

    Args:
        template_name: Name of the template

    Returns:
        str: AI evaluation template string
    """
    templates = {
        "default": AI_template_default,
    }
    if template_name not in templates:
        raise MetricException(f"Unknown template name: {template_name}")
    return templates[template_name]


class AIEvaluator:
    def __init__(self, args):
        """
        Initialize the AI evaluator with the given arguments.

        Args:
            args: Configuration arguments
        """
        self.args = args
        self.eval_model = args.infer_args.AI_eval_model
        self.client = OpenAI(
            api_key=args.infer_args.openai_api_key,
            base_url=args.infer_args.openai_base_url,
            max_retries=args.infer_args.openai_max_retries,
            timeout=args.infer_args.openai_timeout,
        )

    def evaluate_response(
        self, formatted_evaluation: str, model: str
    ) -> Tuple[float, str]:
        """
        Evaluate the response using the AI model.

        Args:
            formatted_evaluation: The formatted evaluation string
            model: The AI model to use for evaluation

        Returns:
            Tuple[float, str]: The score and the evaluation result
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and precise assistant for checking the quality of the answer.",
                    },
                    {"role": "user", "content": formatted_evaluation},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            eval_result = response.choices[0].message.content

            score = float(eval_result.split("score:")[-1].split()[0].split("/")[0])
            return score, eval_result
        except Exception as e:
            logger.exception(f"Error during AI evaluation: {e}")
            return 0.0, ""
