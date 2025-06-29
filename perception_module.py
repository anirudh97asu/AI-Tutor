import logging
from typing import List, Dict, Any
from google import genai
import os
from dotenv import load_dotenv
import asyncio
from concurrent.futures import TimeoutError
import json

load_dotenv()

class Perception:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.logger = logging.getLogger('perception')

    async def generate_with_timeout(self, prompt: str, timeout: int = 10) -> str:
        
        loop = asyncio.get_event_loop()
        
        response = await asyncio.wait_for(
            loop.run_in_executor(None, 
                                 lambda: self.client.models.generate_content
                                 (model="gemini-2.0-flash",
                                  contents=prompt)),
            timeout=timeout
        )

        return response.text.strip()
    

    def perception_prompt(self):
        return """You are an expert tutor specializing in:

                    Data Science: Statistics, data analysis, visualization, and data preprocessing
                    Machine Learning: Supervised/unsupervised learning, model selection, evaluation metrics
                    Deep Learning: Neural networks, architectures (CNN, RNN, Transformers), training techniques
                    GPU Programming: CUDA, parallel computing, performance optimization for ML workloads

                **Instructions:**
                    1. If the query is unrelated to these domains, politely decline and suggest seeking help elsewhere.
                    2. If the query is relevant, provide:
                    - Brief summary of what the user is asking
                    - Key concepts involved
                    - Important sub-topics to explore
                    - Suggested focus areas for comprehensive understanding

                Keep responses concise and educational."""

    async def run_perception_module(self, query):
        
        final_prompt = ""
        
        system_prompt = self.perception_prompt()
        
        final_prompt += system_prompt + "\n\n" + "User_Query: " + "\n\n" + query

        response = await self.generate_with_timeout(final_prompt)

        return response
        