import gradio as gr
from google import genai
import numpy as np
from typing import List, Dict, Any
import json
import logging
from dotenv import load_dotenv
from datetime import datetime
from memory_module import Memory
from perception_module import Perception
import os
import asyncio
import sys
# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Import your RAG backend
from rag_backend import Doc_Processor
processor = Doc_Processor()

# Global variables to maintain state across the session
global_memory = None
global_perception = None
global_parser = None

class GeminiParser:
    def __init__(self, perception_module, memory_module, processor, client):
        self.perception_module = perception_module
        self.memory_module = memory_module
        self.processor = processor
        self.client = client
        self.max_conversation_iteration = 7
   
    def log(self, level: str, message: str) -> None:
        sys.stderr.write(f"{level}: {message}\n")
        sys.stderr.flush()

    def build_socratic_prompt(self):
        """Build the Socratic prompting system prompt"""
        return """
        
        You are a distinguished AI tutor specializing in Data Science, encompassing Statistics, Machine Learning, Python programming, Data Visualization, and Deep Learning. Your students include bootcamp participants and industry professionals seeking to advance their expertise.
        
            - **Core Approach:** Employ the Socratic method as your primary pedagogical framework:
            - **Question-Driven Learning:** Pose thoughtful, guiding questions rather than providing immediate solutions
            - **Reflective Engagement:** Encourage students to articulate their reasoning, challenge assumptions, and explore conceptual connections
            - **Scaffolded Support:** When intervention is necessary, offer strategic hints, illustrative examples, or accessible analogies—reserve complete solutions for exceptional circumstances only

        Communication Standards

            **Professional Demeanor:** Maintain consistently respectful, formal, and supportive discourse that befits academic excellence.
            **Intellectual Development:** Prioritize fostering genuine curiosity and analytical thinking over passive knowledge absorption.

        Instructional Sequence

            **For Unfamiliar Topics:**
                1. Provide conceptual primer and ensure that you provide a good walkthrough about 
                the "what", "when", "why" and "how" questions about the topic asked by the user.
                2. Present engaging, thought-provoking questions about the inutuition and its mathematical details to stimulate active participation
                3. Guide deeper exploration of theoretical frameworks, practical applications, inherent advantages, and potential limitations

        **Ultimate Objectives:** 
            1. Always ensure that your answer aligns with the facts from the knowledge base retrieved and it follows a flow for the given user-query.
            2. Facilitate independent discovery of insights while ensuring students feel intellectually challenged, professionally supported, and academically empowered throughout their learning journey.         
        
        """
   
    def generate_gemini_response(self, client, prompt):
        """Generate content with a timeout"""
        print("Starting LLM generation...")
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            print("LLM generation completed")
            return response
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            raise
            
    def build_perception_query(self, query):
        system_prompt = self.perception_module.perception_prompt()
        final_query = ""
        final_query += "System Prompt: " + "\n\n" + system_prompt + "\n\n"
        if self.memory_module.state.perception_history != [] and self.memory_module.state.user_queries != []:
            for index, perception in enumerate(self.memory_module.state.perception_history):
                final_query += f"User Query at Iteration {index + 1}: " + "\n" + self.memory_module.state.user_queries[index] + "\n"
                final_query += f"Perception at Iteration {index + 1}: " + "\n" + perception + "\n"
            final_query += "\n\n" + "Based on the above series of user-queries and perception provided, try to understand \
                               what the user expects from the new query below and provide your final perception" + "\n"
             
            final_query += "New User_Query: " + "\n\n" + query
        
        else:

            final_query += "User_Query: " + "\n\n" + query
        
        self.log("INFO", final_query)
        return final_query
            
    def build_gemini_conversation(self, knowledge_chunks, query):
        socratic_prompt = self.build_socratic_prompt()
        final_query = ""
        final_query += "System Prompt: " + "\n\n" + socratic_prompt + "\n\n"
        if self.memory_module.state.conversation_history != [] and self.memory_module.state.user_queries != []:
           
            final_query += "Conversation_History: " + "\n\n"
            for index, conversation in enumerate(self.memory_module.state.conversation_history):
                final_query += f"User Query at Iteration {index + 1}: " + "\n" + self.memory_module.state.user_queries[index] + "\n"
                final_query += f"Response at Iteration {index + 1}: " + "\n" + conversation + "\n"
            final_query += "Knowledge base: " + "\n\n"
            for chunk in knowledge_chunks:
                final_query += str(chunk) + "\n"
            final_query += "\n" + "Based on the above series of contexts of user-queries and responses and also rich knowledge base provided, try to understand \
                               what the user expects from the new query below and provide your new response" + "\n"
             
            final_query += "New User_Query: " + "\n\n" + query
           
            return final_query
       
        else:
            
            final_query += "Knowledge base: " + "\n\n"
            for chunk in knowledge_chunks:
                final_query += str(chunk) + "\n"
            final_query += "\n" + "New User_Query: " + "\n\n" + query

            self.log("INFO: GEMINI CONVERSATION QUERY", final_query)

            return final_query
   
    def update_memory(self, perception_response, final_response, query):
        self.memory_module.store(perception_response, final_response, query)
        return
       
    async def run_gemini_module(self, query):
        if self.memory_module.state.iteration >= self.max_conversation_iteration:
            return "Maximum Context Length has been hit. Please start a new conversation altogether."
        perception_query = self.build_perception_query(query)
        perception_response = await self.perception_module.generate_with_timeout(perception_query)
        knowledge_chunks = self.processor.search_documents(perception_response, k=3)
        
        self.log("INFO" + "Knowledge_Chunks", knowledge_chunks)

        final_query = self.build_gemini_conversation(knowledge_chunks, query)
        final_response = self.generate_gemini_response(self.client, final_query)
        
        # Extract text from the response before storing in memory
        if hasattr(final_response, 'text'):
            response_text = final_response.text
        elif hasattr(final_response, 'candidates') and final_response.candidates:
            response_text = final_response.candidates[0].content.parts[0].text
        else:
            response_text = str(final_response)
            
        # Update the memory with extracted text
        self.update_memory(perception_response, response_text, query)
       
        return final_response

def initialize_session():
    """Initialize or reset the conversation session"""
    global global_memory, global_perception, global_parser
    if global_memory is not None:
        global_memory.reset_memory()
   
    global_memory = Memory()
    global_perception = Perception()
    global_parser = GeminiParser(
        perception_module=global_perception,
        memory_module=global_memory,
        processor=processor,
        client=client
    )

def format_conversation_history():
    """Format the conversation history for display"""
    if not global_memory or not global_memory.state.user_queries:
        return "No conversation history yet. Start by asking a question!"
   
    history_text = "📚 **Conversation History**\n\n"
   
    for i, (query, response) in enumerate(zip(global_memory.state.user_queries, global_memory.state.conversation_history)):
        history_text += f"**🔄 Round {i + 1}**\n\n"
        history_text += f"**👤 You:** {query}\n\n"
        history_text += f"**🤖 AI Tutor:** {response}\n\n"
        history_text += "---\n\n"
   
    return history_text

def process_query(user_query, history_display):
    """
    Process user query and return updated conversation
    """
    global global_parser, global_memory
   
    if not user_query.strip():
        return "", history_display, gr.update(visible=False), gr.update(interactive=True)
   
    try:
        # Check if we need to initialize the session
        if global_parser is None:
            initialize_session()
       
        # Get response from Gemini - Note: This needs to be handled differently since it's async
        # For now, using asyncio.run() but you might want to restructure this
        gemini_response = asyncio.run(global_parser.run_gemini_module(user_query))
       
        # Check if we hit the context limit
        if isinstance(gemini_response, str) and "Maximum Context Length has been hit" in gemini_response:
            # Show refresh notification
            return (
                "",  # Clear current response
                history_display,  # Keep current history
                gr.update(visible=True),  # Show refresh button
                gr.update(interactive=False)  # Disable input
            )
       
        # Extract text from response
        if hasattr(gemini_response, 'text'):
            response_text = gemini_response.text
        elif hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            response_text = gemini_response.candidates[0].content.parts[0].text
        else:
            response_text = str(gemini_response)
       
        # Update conversation history display
        updated_history = format_conversation_history()
       
        return (
            response_text,  # Current response
            updated_history,  # Updated history
            gr.update(visible=False),  # Hide refresh button
            gr.update(interactive=True)  # Keep input enabled
        )
       
    except Exception as e:
        error_msg = f"An error occurred while processing your query: {str(e)}"
        print(error_msg)
        return (
            error_msg,
            history_display,
            gr.update(visible=False),
            gr.update(interactive=True)
        )

def refresh_conversation():
    """Reset the conversation and start fresh"""
    initialize_session()
    return (
        "",  # Clear current response
        "No conversation history yet. Start by asking a question!",  # Reset history
        gr.update(visible=False),  # Hide refresh button
        gr.update(interactive=True),  # Enable input
        ""  # Clear input box
    )

def create_gradio_interface():
    """
    Create and configure the conversational Gradio interface
    """
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    .query-box {
        font-size: 16px !important;
    }
    .response-box {
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    .history-box {
        font-size: 13px !important;
        line-height: 1.5 !important;
        background-color: #f8f9fa !important;
    }
    .refresh-warning {
        background-color: #fff3cd !important;
        border: 1px solid #ffeaa7 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    .conversation-counter {
        background-color: #e3f2fd !important;
        padding: 10px !important;
        border-radius: 8px !important;
        text-align: center !important;
        margin-bottom: 15px !important;
    }
    """
   
    with gr.Blocks(css=css, title="Socratic AI Assistant") as interface:
        gr.HTML("<div class='header'><h1>🤔 Socratic AI Assistant</h1><p>Discover knowledge through guided questioning and reasoning</p></div>")
       
        # Conversation counter
        # with gr.Row():
        #     with gr.Column():
        #         conversation_info = gr.HTML(
        #             "<div class='conversation-counter'>💬 <strong>Conversation Rounds Remaining: 3</strong><br><small>You can have up to 3 exchanges before starting a new conversation</small></div>"
        #         )
       
        with gr.Row():
            # Left column - Input and current response
            with gr.Column(scale=1):
                user_input = gr.Textbox(
                    label="💭 Your Question or Follow-up",
                    placeholder="Enter your question or follow-up (e.g., 'Explain decision trees' or 'Can you give me a hint about ensemble methods?')",
                    lines=3,
                    elem_classes=["query-box"]
                )
               
                with gr.Row():
                    submit_btn = gr.Button("🚀 Ask Question", variant="primary", size="lg")
                    refresh_btn = gr.Button("🔄 Start New Conversation", variant="secondary", size="lg", visible=False)
               
                # Warning message for context limit
                context_warning = gr.HTML(
                    "<div class='refresh-warning' style='display: none;'>"
                    "⚠️ <strong>Context Limit Reached!</strong><br>"
                    "You've completed 3 rounds of conversation. Click 'Start New Conversation' to continue learning with a fresh context."
                    "</div>",
                    visible=False
                )
               
                current_response = gr.Textbox(
                    label="🤖 AI Tutor Response",
                    lines=12,
                    max_lines=20,
                    elem_classes=["response-box"],
                    show_copy_button=True,
                    interactive=False
                )
               
                gr.Markdown("""
                ### 💡 How to use:
                1. **Ask your initial question** about any Data Science topic
                2. **Engage with follow-ups** - the AI will guide you with questions
                3. **Think and respond** - this is Socratic learning in action!
                4. **Continue the dialogue** for up to 3 rounds
                5. **Start fresh** when you reach the limit or want a new topic
               
                ### 📚 Example Topics:
                - Machine Learning algorithms and techniques
                - Statistics and data analysis methods  
                - Python programming for data science
                - Database design and optimization
                - Deep learning and neural networks
                """)
           
            # Right column - Conversation history
            with gr.Column(scale=1):
                conversation_history = gr.Textbox(
                    label="📜 Conversation History",
                    value="No conversation history yet. Start by asking a question!",
                    lines=25,
                    max_lines=30,
                    elem_classes=["history-box"],
                    interactive=False,
                    show_copy_button=True
                )
       
        # Event handlers
        submit_btn.click(
            fn=process_query,
            inputs=[user_input, conversation_history],
            outputs=[current_response, conversation_history, context_warning, user_input],
            show_progress=True
        ).then(
            lambda: "",  # Clear input after submission
            outputs=[user_input]
        )
       
        user_input.submit(
            fn=process_query,
            inputs=[user_input, conversation_history],
            outputs=[current_response, conversation_history, context_warning, user_input],
            show_progress=True
        ).then(
            lambda: "",  # Clear input after submission
            outputs=[user_input]
        )
       
        refresh_btn.click(
            fn=refresh_conversation,
            outputs=[current_response, conversation_history, context_warning, user_input, user_input],
            show_progress=True
        )
       
        # Example inputs
        gr.Examples(
            examples=[
                ["What are decision trees and how do they work in machine learning?"],
                ["I'm confused about neural networks. Can you help me understand the basics?"],
                ["Explain the difference between supervised and unsupervised learning."],
                ["How does gradient descent optimization work?"],
                ["What makes ensemble methods effective in machine learning?"],
                ["Can you walk me through the concept of overfitting?"]
            ],
            inputs=[user_input],
            label="🎯 Quick Start Examples",
            cache_examples=False
        )
   
    return interface

if __name__ == "__main__":
    # Check if API key is available
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please set your Gemini API key in your .env file or environment variables.")
        exit(1)
   
    # Initialize the session
    initialize_session()
   
    # Create and launch the interface
    interface = create_gradio_interface()
   
    # Launch with configuration
    interface.launch(
        share=True,  # Set to True if you want to create a public link
        server_name="0.0.0.0",  # Allow access from other devices on network
        server_port=7860,  # Default Gradio port
        show_error=True,
        inbrowser=True  # Automatically open in browser
    )