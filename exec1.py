#!/usr/bin/python3
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import shutil
from pathlib import Path

# Base Intent Handler
class IntentHandler(ABC):
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the intent and return results"""
        pass

# File Operations Handler
class FileOperationsHandler(IntentHandler):
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        operation = params.get('operation')  # create, delete, move, copy, rename
        file_path = params.get('file_path')

        try:
            if operation == 'create':
                content = params.get('content', '')
                Path(file_path).write_text(content)
                return {'status': 'success', 'message': f'Created {file_path}'}

            elif operation == 'delete':
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                return {'status': 'success', 'message': f'Deleted {file_path}'}

            elif operation == 'move':
                destination = params.get('destination')
                shutil.move(file_path, destination)
                return {'status': 'success', 'message': f'Moved {file_path} to {destination}'}

            elif operation == 'copy':
                destination = params.get('destination')
                if os.path.isfile(file_path):
                    shutil.copy2(file_path, destination)
                else:
                    shutil.copytree(file_path, destination)
                return {'status': 'success', 'message': f'Copied {file_path} to {destination}'}

            elif operation == 'rename':
                new_name = params.get('new_name')
                new_path = Path(file_path).parent / new_name
                os.rename(file_path, new_path)
                return {'status': 'success', 'message': f'Renamed to {new_name}'}

            else:
                return {'status': 'error', 'message': f'Unknown operation: {operation}'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# Code Writing Handler
class CodeWritingHandler(IntentHandler):
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code_description = params.get('description')
        language = params.get('language', 'python')
        output_file = params.get('output_file')

        try:
            # You can integrate with an LLM API here (OpenAI, Anthropic, etc.)
            # For now, here's a template approach

            code = self._generate_code(code_description, language)

            if output_file:
                Path(output_file).write_text(code)
                return {
                    'status': 'success',
                    'message': f'Code written to {output_file}',
                    'code': code
                }
            else:
                return {
                    'status': 'success',
                    'message': 'Code generated',
                    'code': code
                }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _generate_code(self, description: str, language: str) -> str:
        """
        Integrate with your preferred LLM API here
        Example with Anthropic API:
        """
        # import anthropic
        # client = anthropic.Anthropic(api_key="your-key")
        # message = client.messages.create(
        #     model="claude-sonnet-4-20250514",
        #     max_tokens=1024,
        #     messages=[{
        #         "role": "user",
        #         "content": f"Write {language} code for: {description}"
        #     }]
        # )
        # return message.content[0].text

        # Placeholder for demonstration
        return f"# Generated {language} code\n# TODO: {description}\n"

# Text Processing Handler
class TextProcessingHandler(IntentHandler):
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        operation = params.get('operation')  # summarize, translate, extract, analyze
        text = params.get('text')

        try:
            if operation == 'summarize':
                summary = self._summarize_text(text, params.get('max_length', 100))
                return {
                    'status': 'success',
                    'operation': 'summarize',
                    'result': summary
                }

            elif operation == 'extract':
                entity_type = params.get('entity_type', 'keywords')
                extracted = self._extract_entities(text, entity_type)
                return {
                    'status': 'success',
                    'operation': 'extract',
                    'result': extracted
                }

            elif operation == 'analyze':
                analysis = self._analyze_sentiment(text)
                return {
                    'status': 'success',
                    'operation': 'analyze',
                    'result': analysis
                }

            else:
                return {'status': 'error', 'message': f'Unknown operation: {operation}'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _summarize_text(self, text: str, max_length: int) -> str:
        """Integrate with LLM or use libraries like sumy, transformers"""
        # Simple extractive summary as placeholder
        sentences = text.split('.')
        return '. '.join(sentences[:3]) + '.'

    def _extract_entities(self, text: str, entity_type: str) -> list:
        """Use spaCy, NLTK, or LLM for entity extraction"""
        # Placeholder
        return ['entity1', 'entity2']

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Use transformers, TextBlob, or LLM for sentiment analysis"""
        # Placeholder
        return {'sentiment': 'neutral', 'confidence': 0.5}

# Intent Executor (Main Dispatcher)
class IntentExecutor:
    def __init__(self):
        self.handlers: Dict[str, IntentHandler] = {
            'file_operations': FileOperationsHandler(),
            'write_code': CodeWritingHandler(),
            'text_processing': TextProcessingHandler(),
        }

    def register_handler(self, intent_type: str, handler: IntentHandler):
        """Register a new intent handler"""
        self.handlers[intent_type] = handler

    def execute(self, intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an intent with given parameters"""
        handler = self.handlers.get(intent)

        if not handler:
            return {
                'status': 'error',
                'message': f'No handler registered for intent: {intent}'
            }

        try:
            result = handler.execute(params)
            return result
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error executing {intent}: {str(e)}'
            }

# Usage Example
if __name__ == "__main__":
    executor = IntentExecutor()

    # Example 1: File Operations
    result = executor.execute('file_operations', {
        'operation': 'create',
        'file_path': 'test.txt',
        'content': 'Hello from voice note!'
    })
    print(result)

    # Example 2: Code Writing
    result = executor.execute('write_code', {
        'description': 'function to calculate factorial',
        'language': 'python',
        'output_file': 'factorial.py'
    })
    print(result)

    # Example 3: Text Processing
    result = executor.execute('text_processing', {
        'operation': 'summarize',
        'text': 'Long text here...',
        'max_length': 50
    })
    print(result)
