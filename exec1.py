#!/usr/bin/python3
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import shutil
from pathlib import Path
import re
import json

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
        
        # Ensure file is in secure folder
        secure_folder = "/home/bosons/interview/"
        if file_path and not file_path.startswith(secure_folder):
            file_path = os.path.join(secure_folder, os.path.basename(file_path))

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
                if destination and not destination.startswith(secure_folder):
                    destination = os.path.join(secure_folder, os.path.basename(destination))
                shutil.move(file_path, destination)
                return {'status': 'success', 'message': f'Moved {file_path} to {destination}'}

            elif operation == 'copy':
                destination = params.get('destination')
                if destination and not destination.startswith(secure_folder):
                    destination = os.path.join(secure_folder, os.path.basename(destination))
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
        """
        Handles both:
        - Writing provided code to a file
        - Generating code from description
        """
        filename = params.get('filename') or params.get('file_path') or params.get('output_file')
        code_body = params.get('body') or params.get('code')
        description = params.get('description')
        language = params.get('language', 'python')
        
        # Ensure file is in secure folder
        secure_folder = "/home/bosons/interview/"
        if filename and not filename.startswith(secure_folder):
            filename = os.path.join(secure_folder, os.path.basename(filename))

        try:
            # If code is provided, use it directly
            if code_body:
                code = self._clean_code(code_body)
            # Otherwise generate from description (placeholder)
            elif description:
                code = self._generate_code(description, language)
            else:
                return {'status': 'error', 'message': 'No code body or description provided'}

            if filename:
                # Ensure directory exists
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                Path(filename).write_text(code)
                # Make executable if it's a script
                if filename.endswith('.py') or filename.endswith('.sh'):
                    os.chmod(filename, 0o755)
                return {
                    'status': 'success',
                    'message': f'Code written to {filename}',
                    'file_path': filename,
                    'code': code
                }
            else:
                return {
                    'status': 'success',
                    'message': 'Code generated (no output file specified)',
                    'code': code
                }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _clean_code(self, code_body: str) -> str:
        """
        Clean up code that might be in pseudo-format
        e.g., "[for i in range(1,7): print(i); exit()]"
        """
        code = str(code_body).strip()
        
        # Remove outer brackets - keep checking until no more outer brackets
        while code.startswith('[') and code.endswith(']'):
            # Make sure we're not removing array brackets that are part of the code
            # Check if this is truly wrapping the whole expression
            bracket_count = 0
            is_wrapper = True
            for i, char in enumerate(code[1:-1], 1):  # Skip first [, check rest
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count < 0:  # Found matching ] before the end
                        is_wrapper = False
                        break
            
            if is_wrapper:
                code = code[1:-1].strip()
            else:
                break
        
        # Replace semicolons with newlines for Python
        if ';' in code:
            code = code.replace('; ', '\n')
            code = code.replace(';', '\n')
        
        # Add proper indentation and formatting
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Decrease indent for certain keywords
            if line.startswith(('else:', 'elif', 'except:', 'finally:')):
                indent_level = max(0, indent_level - 1)
            
            formatted_lines.append('    ' * indent_level + line)
            
            # Increase indent after colon
            if line.endswith(':'):
                indent_level += 1
        
        return '\n'.join(formatted_lines) + '\n'

    def _generate_code(self, description: str, language: str) -> str:
        """
        Placeholder for LLM code generation
        """
        return f"# Generated {language} code\n# TODO: {description}\n"

# Text Processing Handler
class TextProcessingHandler(IntentHandler):
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        operation = params.get('operation', 'summarize')
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
        return ['entity1', 'entity2']

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Use transformers, TextBlob, or LLM for sentiment analysis"""
        return {'sentiment': 'neutral', 'confidence': 0.5}

# Intent Executor (Main Dispatcher)
class IntentExecutor:
    def __init__(self):
        self.handlers: Dict[str, IntentHandler] = {
            'file_operations': FileOperationsHandler(),
            'write_code': CodeWritingHandler(),
            'text_processing': TextProcessingHandler(),
        }
        
        # Map natural language intents to handler names
        self.intent_mapping = {
            'create a file': 'file_operations',
            'create file': 'file_operations',
            'delete file': 'file_operations',
            'delete a file': 'file_operations',
            'move file': 'file_operations',
            'copy file': 'file_operations',
            'rename file': 'file_operations',
            
            'write code': 'write_code',
            'write code to a new file': 'write_code',
            'write code to an existing file': 'write_code',
            'create script': 'write_code',
            'generate code': 'write_code',
            
            'summarize': 'text_processing',
            'summarize text': 'text_processing',
            'extract': 'text_processing',
            'analyze': 'text_processing',
        }

    def register_handler(self, intent_type: str, handler: IntentHandler):
        """Register a new intent handler"""
        self.handlers[intent_type] = handler

    def _parse_params(self, params) -> Dict[str, Any]:
        """
        Parse params which might be:
        1. Already a dict
        2. A JSON string
        3. A string like "[filename:hello.py, body:[...]]"
        """
        if isinstance(params, dict):
            return params
        
        if isinstance(params, str):
            # Try parsing as JSON first
            try:
                return json.loads(params)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Parse custom format: "[filename:hello.py, body:[...]]"
            parsed = {}
            params = params.strip()
            
            # Remove outer brackets if present
            if params.startswith('[') and params.endswith(']'):
                params = params[1:-1]
            
            # Find key:value pairs
            # Handle nested brackets for body
            current_key = None
            current_value = []
            bracket_depth = 0
            i = 0
            
            while i < len(params):
                char = params[i]
                
                if char == '[':
                    bracket_depth += 1
                    if bracket_depth > 0:
                        current_value.append(char)
                elif char == ']':
                    bracket_depth -= 1
                    if bracket_depth >= 0:
                        current_value.append(char)
                elif char == ',' and bracket_depth == 0:
                    # End of key:value pair
                    if current_key:
                        value_str = ''.join(current_value).strip()
                        # Remove brackets if it's a list representation
                        if value_str.startswith('[') and value_str.endswith(']'):
                            value_str = value_str[1:-1]
                        parsed[current_key] = value_str
                    current_key = None
                    current_value = []
                elif char == ':' and bracket_depth == 0 and not current_key:
                    # Found key
                    current_key = ''.join(current_value).strip()
                    current_value = []
                else:
                    current_value.append(char)
                
                i += 1
            
            # Don't forget the last pair
            if current_key:
                value_str = ''.join(current_value).strip()
                if value_str.startswith('[') and value_str.endswith(']'):
                    value_str = value_str[1:-1]
                parsed[current_key] = value_str
            
            return parsed
        
        return {}

    def _normalize_intent(self, intent: str) -> str:
        """Map natural language intent to handler name"""
        intent_lower = intent.lower().strip()
        
        # Direct match
        if intent_lower in self.handlers:
            return intent_lower
        
        # Check mapping
        if intent_lower in self.intent_mapping:
            return self.intent_mapping[intent_lower]
        
        # Partial match
        for key, handler_name in self.intent_mapping.items():
            if key in intent_lower or intent_lower in key:
                return handler_name
        
        return intent_lower

    def execute(self, intent: str, params) -> Dict[str, Any]:
        """
        Execute an intent with given parameters
        
        Args:
            intent: Natural language intent or handler name
            params: Dict, JSON string, or custom format string
        """
        # Normalize the intent
        normalized_intent = self._normalize_intent(intent)
        
        # Parse params
        parsed_params = self._parse_params(params)
        
        # Get handler
        handler = self.handlers.get(normalized_intent)
        
        if not handler:
            return {
                'status': 'error',
                'message': f'No handler registered for intent: {intent} (normalized: {normalized_intent})'
            }

        try:
            print(f"\n=== Executing Intent ===")
            print(f"Original intent: {intent}")
            print(f"Normalized intent: {normalized_intent}")
            print(f"Parsed params: {parsed_params}")
            print(f"========================\n")
            
            result = handler.execute(parsed_params)
            return result
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error executing {intent}: {str(e)}'
            }

# Usage Example
if __name__ == "__main__":
    executor = IntentExecutor()

    # Test with your actual LLM output format
    print("Test 1: Your actual LLM output")
    result = executor.execute(
        'write code to a new file',
        '[filename:hello.py, body:[for i in range(1,7): print(i); exit()]'
    )
    print(f"Result: {result}\n")
    
    # Test reading the created file
    if result['status'] == 'success':
        print("Generated code:")
        with open(result['file_path'], 'r') as f:
            print(f.read())

    print("\n" + "="*50 + "\n")

    # Test with dict params
    print("Test 2: Dict params")
    result = executor.execute('write_code', {
        'filename': 'test2.py',
        'body': 'print("Hello World")'
    })
    print(f"Result: {result}\n")

    print("\n" + "="*50 + "\n")

    # Test file operations
    print("Test 3: File operations")
    result = executor.execute('create a file', {
        'operation': 'create',
        'file_path': 'notes.txt',
        'content': 'This is a test note.'
    })
    print(f"Result: {result}\n")
