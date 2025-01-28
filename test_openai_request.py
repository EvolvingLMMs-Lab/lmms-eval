import unittest
from unittest.mock import patch, Mock
import json
import os
from dotenv import load_dotenv
from lmms_eval.tasks.genai_rqa.utils import get_eval, EVALUATION_SYSTEM_PROMPT
from lmms_eval.tasks.genai_rqa.prompt import EVALUATION_RESPONSE_SCHEMA

load_dotenv()

class TestGetEval(unittest.TestCase):
    def setUp(self):
        self.test_content = """
參考文本: 根據教材，主動學習已被證明能夠提升學生的參與度和知識保留，優於被動學習方法。
問題: 主動學習相比被動學習的主要好處是什麼？
生成的答案: 主動學習比被動學習更能有效地吸引學生。
"""
        self.max_tokens = 500

    @patch('requests.post')
    def test_get_eval_response_handling(self, mock_post):
        # Create response content following the schema
        response_content = {
            "reasoning_steps": "參考資料明確指出，主動學習提升了參與度和知識保留，這支持了生成的答案中提到主動學習更能有效地吸引學生。",
            "output_comments": "答案準確且符合參考資料中提出的核心概念。",
            "correctness_label": "Perfect",
            "richness_score": 3,
            "richness_comments": "答案雖然正確，但僅包含基本內容。",
            "completeness_score": 4,
            "completeness_comments": "答案覆蓋了問題的主要關鍵點。"
        }
        
        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        
        # Set up the json method to return proper structure
        response_json = {
            "choices": [{
                "message": {
                    "content": json.dumps(response_content)
                }
            }],
            "model": "gpt-4o-mini"
        }
        mock_response.json.return_value = response_json
        
        # Set up the response object attributes
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps(response_content)
        
        mock_post.return_value = mock_response

        # Call get_eval
        content, model = get_eval(self.test_content, self.max_tokens)
        
        # Verify the request payload
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs['json']
        
        # Check response_format matches the schema from prompt.py
        self.assertEqual(
            payload.get('response_format'),
            EVALUATION_RESPONSE_SCHEMA,
            "response_format should match EVALUATION_RESPONSE_SCHEMA from prompt.py"
        )
        
        # Print actual request payload for debugging
        print("\nActual Request Payload:")
        print(json.dumps(payload, indent=2))
        
        # Print response handling details
        print("\nMocked Response:")
        print(json.dumps(mock_response.json(), indent=2))
        
        # Print parsed content
        print("\nParsed Content:")
        print(json.dumps(content, indent=2))
        
        # Verify content was parsed correctly
        self.assertIsInstance(content, dict)
        self.assertEqual(content["correctness_label"], "Perfect")
        self.assertEqual(content["richness_score"], 3)
        self.assertEqual(content["completeness_score"], 4)
        self.assertEqual(model, "gpt-4o-mini")

if __name__ == '__main__':
    unittest.main()
