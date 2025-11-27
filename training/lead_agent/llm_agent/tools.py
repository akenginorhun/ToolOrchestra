# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------------

import time
import json
from .call_llm import get_llm_response

def call_api(prompt):
    # print(6,'prompt',prompt)
    response = get_llm_response(model='o3',messages=[{'role': 'user','content': prompt}],temperature=1)
    # print(8,'response',response)
    return response

class QueryWriter:

    def __init__(self):
        # self.prompt = ("Documents:\n{documents}\n\nUser question: {user_question}\n\n"
        #                "Based on the documents we have found, write queries to search missing information. Use the following format to output:\n"
        #                "<start>"
        #                "1. query1\n"
        #                "2. query2\n"
        #                "3. ...\n"
        #                "<end>")
        self.prompt = ("Documents:\n{documents}\n\nUser question: {user_question}\n\n"
                        # "(1) Identify the essential problem in the post. (2) Think step by step to reason about what should be included in the relevant documents. (3) Draft an answer")
                        # "Break down the user question. Analyze which subquestion is answered by which document and which subquestion is not answered by any document. Wrap only the first unanswered subquestions within <subquestion> and </subquestion>")
                       "Break down the user question. Based on the documents we have found, write a query to search missing information. Wrap the query within <query> and </query>")
                    # "Analyze what information we have now in documents, write only a sigle query to search a part of missing information. Note that we do not need to find all missing information at this step. Just find some missing information would be fine. Wrap the analysis between <analysis> and </analysis>. Wrap the single query within <query> and </query>")

    def __call__(self, documents, user_question):
        results = []
        call_count = 0
        response = ''
        while results==[] and call_count < 3:
            call_count += 1
            try:
                response = call_api(prompt=self.prompt.format(documents=documents, user_question=user_question))
                results = self.parse_response(response)
                return results
            except:
                # print(f"Fail to parse query writer response",response,'##')
                time.sleep(3)
        # print('query writer count:',call_count)
        # if results==[]:
        #     print(f"Fail to get query writer response",response,'##')
        return results

    # def parse_response(self, response):
    #     queries = response.split("<start>")[1].split("<end>")[0].strip()
    #     if not queries.startswith("1."):
    #         return []
    #     queries = queries[len("1."):].strip()
    #     generated_queries = []
    #     for idx in range(2,5):
    #         components = queries.split(f"\n{idx}.")
    #         cur_query = components[0]
    #         queries = f"\n{idx}.".join(components[1:])
    #         if len(cur_query)>0:
    #             generated_queries.append(cur_query)
    #     return generated_queries

    def parse_response(self, response):
        results = response.split("<query>")[-1].split("</query>")[0].strip()
        return [results]


class AnswerGenerator:

    def __init__(self):
        self.prompt = ("Documents:\n{documents}\n\nUser question: {user_question}\n\n"
                       "Wrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>.")

    def __call__(self, documents, user_question):
        results = ''
        call_count = 0
        response = ''
        while results=='': # and call_count < 3:
            call_count += 1
            try:
                cur_prompt = self.prompt.format(documents=documents, user_question=user_question)
                response = call_api(prompt=cur_prompt)
                results = self.parse_response(response)
                # print('prompt:\n\n',cur_prompt,'\n\n\n')
                # print('response:\n\n',response,'\n\n\n')
                # print('results:\n\n',results,'\n\n\n','='*100)
                # print(86,results)
                return results
            except:
                # print(f"Fail to parse answer generator response",response,'##')
                time.sleep(3)
        # print('answer generator count:',call_count)
        return results

    def parse_response(self, response):
        results = response.split("<answer>")[-1].split("</answer>")[0].strip()
        return results

TOOLS_MAP = {
    'query writer': QueryWriter,
    'answer generator': AnswerGenerator,
}
