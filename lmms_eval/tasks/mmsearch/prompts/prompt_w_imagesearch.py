
stage1_image_search_text_requery_prompt = '''You are a helpful assistant. I am giving you a question including an image, which cannot be solved without external knowledge.
Assume you have access to a search engine (e.g., google). Please raise a query to the search engine to search for what is useful for you to answer the question correctly. You need to consider the characteristics of asking questions to search engines when formulating your questions. 
You are also provided with the search result of the image in the question. You should leverage the image search result to raise the text query.
Here are 3 examples:
Question: Did Zheng Xiuwen wear a knee pad in the women's singles tennis final in 2024 Paris Olympics?
Query to the search engine: Images of Zheng Xiuwen in the women's singles tennis final in 2024 Paris Olympics

Question: When will Apple release iPhone16?
Query to the search engine: iPhone 16 release date 

Question: Who will sing a French song at the Olympic Games closing ceremony?
Query to the search engine: Singers at the Olympic Games closing ceremony, French song

Question: {question} 
The image search result is: {image_search_result}
Query to the search engine (do not involve any explanation): '''

stage2_image_search_text_requery_prompt = '''You are a helpful assistant. I am giving you a question including an image. You are provided with the search result of the image in the question. And you are provided with {brief_result_num} website information related to the question (including the screenshot, snippet and title). 
You should now read the screenshots, snippets and titles of these websites. Select {rerank_num} website that are the most helpful for you to answer the question. Once you select it, the detailed content of them will be provided to help you correctly answer the question.
The question is: {question}
The image search result is: {image_search_result}
The website informations is: 
{website_information}

You should directly output {rerank_num} website's index that can help you most, separated with ',', and enclose each website in angle brackets. The output format should be: <Website Index>. 
An example of the output is: {incontext_example}
Your answer: '''

stage3_image_search_text_requery_prompt = '''You are a helpful assistant. I am giving you a question including an image. You are provided with the search result of the image in the question. And you are provided with {rerank_num} website information related to the question. 
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. When answering questions about dates, use the yyyy-mm-dd format.
3. Answer the question with as few words as you can.

You should now read the information of the website and answer the question.
The website informations is {website_information}
The image search result is: {image_search_result}
The question is: {question}.
Please directly output the answer without any explanation: '''


image_search_text_query_dict = {
        'stage1': stage1_image_search_text_requery_prompt,
        'stage2': stage2_image_search_text_requery_prompt,
        'stage3': stage3_image_search_text_requery_prompt,
}
