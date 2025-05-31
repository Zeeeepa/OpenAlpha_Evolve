"""
Language Detector for automatic programming language identification.

Detects programming languages based on file extensions, content patterns,
and syntax analysis to enable appropriate parsing and analysis.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..core.interfaces import LanguageType


logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detector for programming languages based on various heuristics.
    """
    
    def __init__(self):
        """Initialize the language detector."""
        
        # File extension mappings
        self.extension_mappings = {
            '.py': LanguageType.PYTHON,
            '.pyw': LanguageType.PYTHON,
            '.js': LanguageType.JAVASCRIPT,
            '.jsx': LanguageType.JAVASCRIPT,
            '.mjs': LanguageType.JAVASCRIPT,
            '.ts': LanguageType.TYPESCRIPT,
            '.tsx': LanguageType.TYPESCRIPT,
            '.java': LanguageType.JAVA,
            '.cpp': LanguageType.CPP,
            '.cxx': LanguageType.CPP,
            '.cc': LanguageType.CPP,
            '.c++': LanguageType.CPP,
            '.hpp': LanguageType.CPP,
            '.hxx': LanguageType.CPP,
            '.h++': LanguageType.CPP,
            '.c': LanguageType.CPP,  # Treat C as C++ for simplicity
            '.h': LanguageType.CPP,
            '.rs': LanguageType.RUST,
            '.go': LanguageType.GO
        }
        
        # Content-based patterns for language detection
        self.content_patterns = {
            LanguageType.PYTHON: [
                r'^\s*def\s+\w+\s*\(',
                r'^\s*class\s+\w+',
                r'^\s*import\s+\w+',
                r'^\s*from\s+\w+\s+import',
                r'if\s+__name__\s*==\s*["\']__main__["\']',
                r'^\s*#.*python',
                r'print\s*\(',
                r'^\s*@\w+'  # decorators
            ],
            LanguageType.JAVASCRIPT: [
                r'function\s+\w+\s*\(',
                r'var\s+\w+\s*=',
                r'let\s+\w+\s*=',
                r'const\s+\w+\s*=',
                r'console\.log\s*\(',
                r'require\s*\(["\']',
                r'module\.exports',
                r'=>',  # arrow functions
                r'document\.',
                r'window\.'
            ],
            LanguageType.TYPESCRIPT: [
                r'interface\s+\w+',
                r'type\s+\w+\s*=',
                r':\s*\w+\s*=',  # type annotations
                r'<\w+>',  # generics
                r'export\s+interface',
                r'export\s+type',
                r'import\s+.*from\s+["\'].*["\']'
            ],
            LanguageType.JAVA: [
                r'public\s+class\s+\w+',
                r'private\s+\w+\s+\w+',
                r'public\s+static\s+void\s+main',
                r'import\s+java\.',
                r'package\s+\w+',
                r'System\.out\.println',
                r'@Override',
                r'extends\s+\w+',
                r'implements\s+\w+'
            ],
            LanguageType.CPP: [
                r'#include\s*<\w+>',
                r'#include\s*"\w+"',
                r'using\s+namespace\s+std',
                r'int\s+main\s*\(',
                r'std::\w+',
                r'cout\s*<<',
                r'cin\s*>>',
                r'class\s+\w+\s*{',
                r'template\s*<'
            ],
            LanguageType.RUST: [
                r'fn\s+\w+\s*\(',
                r'let\s+\w+\s*=',
                r'let\s+mut\s+\w+',
                r'use\s+\w+',
                r'mod\s+\w+',
                r'impl\s+\w+',
                r'struct\s+\w+',
                r'enum\s+\w+',
                r'match\s+\w+',
                r'println!\s*\('
            ],
            LanguageType.GO: [
                r'package\s+\w+',
                r'import\s+\(',
                r'func\s+\w+\s*\(',
                r'func\s+main\s*\(',
                r'fmt\.Print',
                r'type\s+\w+\s+struct',
                r'go\s+\w+\s*\(',
                r'chan\s+\w+',
                r'defer\s+\w+'
            ]
        }
        
        # Shebang patterns
        self.shebang_patterns = {
            r'#!/usr/bin/env python': LanguageType.PYTHON,
            r'#!/usr/bin/python': LanguageType.PYTHON,
            r'#!/usr/bin/env node': LanguageType.JAVASCRIPT,
            r'#!/usr/bin/node': LanguageType.JAVASCRIPT
        }
        
        # Common keywords for each language (for scoring)
        self.language_keywords = {
            LanguageType.PYTHON: {
                'def', 'class', 'import', 'from', 'if', 'elif', 'else', 'for', 'while',
                'try', 'except', 'finally', 'with', 'as', 'lambda', 'yield', 'return',
                'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False', 'self'
            },
            LanguageType.JAVASCRIPT: {
                'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while',
                'do', 'switch', 'case', 'break', 'continue', 'return', 'try', 'catch',
                'finally', 'throw', 'new', 'this', 'prototype', 'null', 'undefined',
                'true', 'false', 'typeof', 'instanceof'
            },
            LanguageType.TYPESCRIPT: {
                'interface', 'type', 'enum', 'namespace', 'module', 'declare',
                'export', 'import', 'extends', 'implements', 'public', 'private',
                'protected', 'readonly', 'static', 'abstract', 'async', 'await'
            },
            LanguageType.JAVA: {
                'public', 'private', 'protected', 'static', 'final', 'abstract',
                'class', 'interface', 'extends', 'implements', 'package', 'import',
                'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break',
                'continue', 'return', 'try', 'catch', 'finally', 'throw', 'throws',
                'new', 'this', 'super', 'null', 'true', 'false', 'void', 'int',
                'String', 'boolean'
            },
            LanguageType.CPP: {
                'int', 'char', 'float', 'double', 'void', 'bool', 'class', 'struct',
                'enum', 'union', 'namespace', 'using', 'template', 'typename',
                'public', 'private', 'protected', 'virtual', 'static', 'const',
                'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break',
                'continue', 'return', 'try', 'catch', 'throw', 'new', 'delete',
                'this', 'nullptr', 'true', 'false'
            },
            LanguageType.RUST: {
                'fn', 'let', 'mut', 'const', 'static', 'struct', 'enum', 'impl',
                'trait', 'mod', 'use', 'pub', 'crate', 'super', 'self', 'Self',
                'if', 'else', 'match', 'for', 'while', 'loop', 'break', 'continue',
                'return', 'move', 'ref', 'true', 'false', 'Some', 'None', 'Ok', 'Err'
            },
            LanguageType.GO: {
                'package', 'import', 'func', 'var', 'const', 'type', 'struct',
                'interface', 'map', 'chan', 'go', 'defer', 'if', 'else', 'for',
                'switch', 'case', 'break', 'continue', 'return', 'range', 'select',
                'true', 'false', 'nil', 'make', 'new', 'len', 'cap', 'append'
            }
        }
        
        logger.debug("Language detector initialized")
    
    def detect_language(self, content: str, file_path: Optional[str] = None) -> LanguageType:
        """
        Detect the programming language of the given content.
        
        Args:
            content: Source code content
            file_path: Optional file path for extension-based detection
            
        Returns:
            Detected language type
        """
        try:
            # Try extension-based detection first
            if file_path:
                ext_language = self._detect_by_extension(file_path)
                if ext_language != LanguageType.UNKNOWN:
                    logger.debug(f"Language detected by extension: {ext_language}")
                    return ext_language
            
            # Try shebang detection
            shebang_language = self._detect_by_shebang(content)
            if shebang_language != LanguageType.UNKNOWN:
                logger.debug(f"Language detected by shebang: {shebang_language}")
                return shebang_language
            
            # Try content-based detection
            content_language = self._detect_by_content(content)
            if content_language != LanguageType.UNKNOWN:
                logger.debug(f"Language detected by content: {content_language}")
                return content_language
            
            # Try keyword-based scoring
            keyword_language = self._detect_by_keywords(content)
            if keyword_language != LanguageType.UNKNOWN:
                logger.debug(f"Language detected by keywords: {keyword_language}")
                return keyword_language
            
            logger.debug("Could not detect language, returning UNKNOWN")
            return LanguageType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Language detection failed", exc_info=e)
            return LanguageType.UNKNOWN
    
    def _detect_by_extension(self, file_path: str) -> LanguageType:
        """Detect language by file extension."""
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            return self.extension_mappings.get(extension, LanguageType.UNKNOWN)
            
        except Exception as e:
            logger.error(f"Extension detection failed", exc_info=e)
            return LanguageType.UNKNOWN
    
    def _detect_by_shebang(self, content: str) -> LanguageType:
        """Detect language by shebang line."""
        try:
            lines = content.split('\n')
            if not lines:
                return LanguageType.UNKNOWN
            
            first_line = lines[0].strip()
            if not first_line.startswith('#!'):
                return LanguageType.UNKNOWN
            
            for pattern, language in self.shebang_patterns.items():
                if re.search(pattern, first_line):
                    return language
            
            return LanguageType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Shebang detection failed", exc_info=e)
            return LanguageType.UNKNOWN
    
    def _detect_by_content(self, content: str) -> LanguageType:
        """Detect language by content patterns."""
        try:
            language_scores = {}
            
            for language, patterns in self.content_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                    score += len(matches)
                
                if score > 0:
                    language_scores[language] = score
            
            if language_scores:
                # Return language with highest score
                best_language = max(language_scores, key=language_scores.get)
                
                # Only return if score is significant
                if language_scores[best_language] >= 2:
                    return best_language
            
            return LanguageType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Content detection failed", exc_info=e)
            return LanguageType.UNKNOWN
    
    def _detect_by_keywords(self, content: str) -> LanguageType:
        """Detect language by keyword frequency."""
        try:
            # Extract words from content
            words = re.findall(r'\b\w+\b', content.lower())
            if not words:
                return LanguageType.UNKNOWN
            
            word_set = set(words)
            language_scores = {}
            
            for language, keywords in self.language_keywords.items():
                # Count keyword matches
                matches = len(word_set.intersection(keywords))
                
                # Calculate score as percentage of keywords found
                if keywords:
                    score = matches / len(keywords)
                    language_scores[language] = score
            
            if language_scores:
                # Return language with highest score
                best_language = max(language_scores, key=language_scores.get)
                
                # Only return if score is significant (at least 10% keyword match)
                if language_scores[best_language] >= 0.1:
                    return best_language
            
            return LanguageType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Keyword detection failed", exc_info=e)
            return LanguageType.UNKNOWN
    
    def get_language_confidence(self, content: str, file_path: Optional[str] = None) -> Dict[LanguageType, float]:
        """
        Get confidence scores for all languages.
        
        Args:
            content: Source code content
            file_path: Optional file path
            
        Returns:
            Dictionary mapping languages to confidence scores (0-1)
        """
        try:
            confidence_scores = {}
            
            # Extension-based confidence
            if file_path:
                ext_language = self._detect_by_extension(file_path)
                if ext_language != LanguageType.UNKNOWN:
                    confidence_scores[ext_language] = confidence_scores.get(ext_language, 0) + 0.4
            
            # Shebang-based confidence
            shebang_language = self._detect_by_shebang(content)
            if shebang_language != LanguageType.UNKNOWN:
                confidence_scores[shebang_language] = confidence_scores.get(shebang_language, 0) + 0.3
            
            # Content pattern confidence
            for language, patterns in self.content_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                    score += len(matches)
                
                if score > 0:
                    # Normalize score (max 0.3)
                    normalized_score = min(score / 10, 0.3)
                    confidence_scores[language] = confidence_scores.get(language, 0) + normalized_score
            
            # Keyword-based confidence
            words = re.findall(r'\b\w+\b', content.lower())
            if words:
                word_set = set(words)
                
                for language, keywords in self.language_keywords.items():
                    matches = len(word_set.intersection(keywords))
                    if keywords and matches > 0:
                        # Normalize score (max 0.3)
                        normalized_score = min((matches / len(keywords)) * 0.3, 0.3)
                        confidence_scores[language] = confidence_scores.get(language, 0) + normalized_score
            
            # Normalize all scores to 0-1 range
            max_score = max(confidence_scores.values()) if confidence_scores else 1
            if max_score > 1:
                confidence_scores = {lang: score / max_score for lang, score in confidence_scores.items()}
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Confidence calculation failed", exc_info=e)
            return {}
    
    def is_supported_language(self, language: LanguageType) -> bool:
        """
        Check if a language is supported for analysis.
        
        Args:
            language: Language type to check
            
        Returns:
            True if supported, False otherwise
        """
        supported_languages = {
            LanguageType.PYTHON,
            LanguageType.JAVASCRIPT,
            LanguageType.TYPESCRIPT,
            LanguageType.JAVA,
            LanguageType.CPP,
            LanguageType.RUST,
            LanguageType.GO
        }
        
        return language in supported_languages
    
    def get_file_type_hint(self, file_path: str) -> Optional[str]:
        """
        Get a hint about the file type based on path patterns.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            File type hint or None
        """
        try:
            path = Path(file_path)
            
            # Check for common patterns
            if 'test' in path.name.lower():
                return 'test_file'
            elif path.name.lower() in ['readme.md', 'readme.txt', 'readme']:
                return 'documentation'
            elif path.name.lower() in ['makefile', 'dockerfile', 'docker-compose.yml']:
                return 'build_file'
            elif path.suffix.lower() in ['.json', '.yaml', '.yml', '.toml', '.ini']:
                return 'config_file'
            elif path.suffix.lower() in ['.md', '.rst', '.txt']:
                return 'documentation'
            elif path.name.lower().startswith('.'):
                return 'hidden_file'
            
            return None
            
        except Exception as e:
            logger.error(f"File type hint failed", exc_info=e)
            return None

