import asyncio
from typing import List, Any
from pydantic import BaseModel


class MemoryState(BaseModel):
    iteration: int
    user_queries: List[str]
    perception_history: List[Any]
    current_response: str
    conversation_history: List[Any]
    

class Memory:
    """The memory module for an LLM to store important facts, results and discoveries."""
    def __init__(self):
        self.state = MemoryState(iteration=0,
                                 user_queries=[],
                                 perception_history=[],
                                 conversation_history=[],
                                 current_response="")
    
    def store(self, perception, iteration_response, user_query):
        self.state.perception_history.append(perception)
        self.state.user_queries.append(user_query)
        self.state.conversation_history.append(iteration_response)
        self.state.current_response = iteration_response
        self.state.iteration += 1
    
    def generate_next_prompt(self, system_prompt, original_query, perception_outline):
        new_prompt = ""
        new_prompt += system_prompt + "\n\n"

        iteration_history = self.state.iteration_history

        new_prompt += f"Original_Query: {original_query}" + "\n\n"

        new_prompt += f"Perception Outline: {perception_outline}" + "\n\n"

        if len(iteration_history):
            new_prompt += "Iteration_history:" + "\n\n"
            for iteration, operation in enumerate(iteration_history):
                new_prompt += f"""Iteration: {iteration},\n
                                  Function call: {operation["name"]},\n
                                  Result: {operation["result"]}\n\n"""
        
        new_prompt += "\n\n" + "Given this detailed conversation history and updated user-quer, try to answer accordingly." + "\n\n"
        
        return new_prompt
        
    def calculate_next_step(self):
        next_query = self.generate_next_prompt(self.system_prompt, iteration_history=[])
        return next_query

    def reset_memory(self):
        self.state = MemoryState(iteration=0,
                                 user_queries=[],
                                 perception_history=[],
                                 current_response="",
                                 conversation_history=[])