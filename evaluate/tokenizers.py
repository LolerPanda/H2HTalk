#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tokenizer module, providing multilingual text tokenization functionality
Supports tokenization for Chinese, Japanese, and English text
"""

import os
import re
import time
from core import timed_print, HAVE_JIEBA, HAVE_FUGASHI

# Create PTBTokenizer cache to avoid repeated initialization
tokenizer_cache = {}

class JiebaTokenizer:
    """
    Use jieba tokenizer as a replacement for PTBTokenizer while maintaining interface compatibility
    """
    def __init__(self):
        timed_print("【JiebaTokenizer】Starting initialization...")
        init_start = time.time()
        import jieba
        self.jieba = jieba
        print("\n" + "#"*60)
        timed_print("【JiebaTokenizer】JiebaTokenizer initialization completed")
        timed_print("【JiebaTokenizer】This is a Chinese-optimized tokenizer that replaces the original PTBTokenizer")
        print("#"*60 + "\n")
        # Record total processed tokens and time
        self.total_tokens = 0
        self.total_time = 0
        self.call_count = 0
        # Create log file
        self._init_log()
        self._log("JiebaTokenizer initialization completed")
        init_time = time.time() - init_start
        timed_print(f"【JiebaTokenizer】Initialization completed (Time taken: {init_time:.2f} seconds)")
    
    def _init_log(self):
        """Initialize log file"""
        import os
        import time
        
        log_start = time.time()
        # Ensure directory exists
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            timed_print(f"【JiebaTokenizer】Created log directory: {log_dir}")
        
        # Fix log file path to ensure it's under logs directory
        self.log_file = os.path.join(log_dir, f"jieba_tokenizer_pid{os.getpid()}.log")
        timed_print(f"【JiebaTokenizer】Created dedicated log file: {self.log_file}")
        
        # Ensure new log file is created for each run
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"--- JiebaTokenizer Log PID:{os.getpid()} Start Time:{time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        # Also record to common log file
        main_log_file = os.path.join(log_dir, "tokenizer_calls.log")
        with open(main_log_file, "a", encoding="utf-8") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] Process {os.getpid()}: JiebaTokenizer log file created at {self.log_file}\n")
        
        log_time = time.time() - log_start
        timed_print(f"【JiebaTokenizer】Log initialization completed (Time taken: {log_time:.2f} seconds)")
    
    def _log(self, message):
        """Write to log file"""
        import os
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] PID:{os.getpid()} - {message}\n")
            
            # Also record key operations to main log
            if "Initialization" in message or "Tokenization result" in message:
                main_log_file = os.path.join("logs", "tokenizer_calls.log")
                with open(main_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] Process {os.getpid()} - {message}\n")
        except Exception as e:
            timed_print(f"Error writing to log: {e}")
    
    def tokenize(self, sentences):
        """
        Tokenize Chinese text while maintaining the same interface as PTBTokenizer
        
        Args:
            sentences: Can be a string or dictionary
            
        Returns:
            Output in the same format as PTBTokenizer
        """
        # Record processing start time
        import time
        import os
        start_time = time.time()
        
        timed_print(f"【JiebaTokenizer】Starting tokenization, input type:{type(sentences)}")
        self._log(f"Starting tokenization, input type:{type(sentences)}")
        
        result = {}
        
        # Process different types of input
        if isinstance(sentences, str):
            # Single string, create a temporary ID
            temp_id = "temp_id"
            sentences = {temp_id: [sentences]}
            self._log(f"Input is string, converted to dictionary:{sentences}")
            timed_print(f"【JiebaTokenizer】Input is string, converted to dictionary format")
        
        # Record input data summary
        keys_count = len(sentences)
        total_sentences = sum(len(sentence_list) for sentence_list in sentences.values())
        self._log(f"Input data: {keys_count} keys, total {total_sentences} sentences")
        timed_print(f"【JiebaTokenizer】Input data: {keys_count} keys, total {total_sentences} sentences")
        
        try:
            # Process dictionary format input
            tokens_count = 0
            for key, sentence_list in sentences.items():
                result[key] = []
                for sentence in sentence_list:
                    if not sentence:
                        tokens = []
                        self._log(f"Key:{key} - Empty sentence")
                        timed_print(f"【JiebaTokenizer】Warning: Key {key} contains empty sentence")
                    else:
                        # Record sentence length
                        sentence_len = len(sentence)
                        # Use jieba for tokenization
                        tokens = list(self.jieba.cut(sentence))
                        tokens_count += len(tokens)
                        self._log(f"Key:{key} - Sentence length:{sentence_len}, Token count:{len(tokens)}")
                    result[key].append(tokens)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.total_tokens += tokens_count
            self.total_time += processing_time
            self.call_count += 1
            
            # Calculate and display performance data
            tokens_per_second = tokens_count / processing_time if processing_time > 0 and tokens_count > 0 else 0
            avg_tokens_per_second = self.total_tokens / self.total_time if self.total_time > 0 and self.total_tokens > 0 else 0
            
            # Record tokenization performance
            perf_message = f"Tokenization result: {tokens_count} tokens, Time taken:{processing_time:.4f} seconds, Speed:{tokens_per_second:.2f}t/s, Total:{self.total_tokens} tokens, Average speed:{avg_tokens_per_second:.2f}t/s, Call count:{self.call_count}"
            self._log(perf_message)
            
            # Print clear tokenizer usage information
            print(f"\n{'+'*60}")
            timed_print(f"【JiebaTokenizer】Processing completed! Processed {tokens_count} tokens, Speed: {tokens_per_second:.2f} tokens/s")
            timed_print(f"【JiebaTokenizer】Total processed: {self.total_tokens} tokens, Average speed: {avg_tokens_per_second:.2f} tokens/s")
            timed_print(f"【JiebaTokenizer】Call count: {self.call_count}")
            print(f"{'+'*60}\n")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during tokenization: {str(e)}"
            self._log(error_msg)
            print(f"\n{'!'*60}")
            timed_print(f"【JiebaTokenizer Error】{error_msg}")
            print(f"{'!'*60}\n")
            # Return empty result instead of raising exception to keep program running
            return {k: [[]] for k in sentences.keys()}

class SimpleTokenizer:
    """Simple tokenizer, used as fallback when other tokenizers are unavailable"""
    
    def tokenize(self, sentences):
        """Simple tokenization, split text by spaces"""
        result = {}
        
        if isinstance(sentences, str):
            temp_id = "temp_id"
            sentences = {temp_id: [sentences]}
        
        for key, sentence_list in sentences.items():
            result[key] = []
            for sentence in sentence_list:
                if not sentence:
                    tokens = []
                else:
                    # Simple space-based tokenization
                    tokens = sentence.lower().split()
                result[key].append(tokens)
        
        return result

def get_tokenizer():
    """
    Get shared tokenizer instance
    Use a single tokenizer instance throughout the evaluation process to avoid repeated initialization
    """
    import os
    
    try:
        # Get current process ID
        pid = os.getpid()
        pid_str = str(pid)
        
        # Check if tokenizer cache exists
        global tokenizer_cache
        if pid in tokenizer_cache:
            reuse_message = f"Process {pid}: Reusing existing tokenizer instance {type(tokenizer_cache[pid]).__name__}"
            timed_print(reuse_message)
            return tokenizer_cache[pid]
        
        # Try to import and use jieba
        try:
            # First try to use custom JiebaTokenizer
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            try:
                import jieba
                jieba_start = time.time()
                tokenizer_cache[pid] = JiebaTokenizer()
                jieba_time = time.time() - jieba_start
                timed_print(f"JiebaTokenizer created successfully! Time taken: {jieba_time:.2f} seconds")
                return tokenizer_cache[pid]
            except ImportError:
                timed_print("Jieba tokenizer library not found, will use PTBTokenizer")
        except Exception as e:
            timed_print(f"Error initializing JiebaTokenizer: {str(e)}")
        
        # If jieba is unavailable, use PTBTokenizer
        try:
            ptb_start = time.time()
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
            tokenizer_cache[pid] = PTBTokenizer()
            ptb_time = time.time() - ptb_start
            timed_print(f"PTBTokenizer creation time: {ptb_time:.2f} seconds")
        except ImportError:
            timed_print("Cannot import PTBTokenizer, will use simple space-based tokenization")
            tokenizer_cache[pid] = SimpleTokenizer()
        
        return tokenizer_cache[pid]
    except Exception as e:
        timed_print(f"Error getting tokenizer: {str(e)}")
        return SimpleTokenizer()  # Return simple tokenizer as fallback

def tokenize_text(text, is_japanese=None):
    """
    Tokenize text, supporting Chinese, Japanese, and English
    
    Args:
        text: Text to be tokenized
        is_japanese: Whether the text is Japanese, if None, auto-detect
        
    Returns:
        List of tokens after tokenization
    """
    # Check if text is Chinese or Japanese
    # if is_japanese is None:
    #     # Simple detection of Chinese or Japanese characters
    #     cjk_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text)
    #     is_japanese = len(cjk_chars) > 0
    
    # Chinese or Japanese processing
    # if is_japanese:
        # Detect Chinese character ratio
        # chinese_chars = re.findall(r'[\u4E00-\u9FFF]', text)
        # is_chinese = len(chinese_chars) > len(cjk_chars) * 0.5  # If Chinese characters exceed half, consider as Chinese
        
    is_chinese=True
    if is_chinese and HAVE_JIEBA:
            # Use jieba tokenizer
        try:
            import jieba
            return list(jieba.cut(text))
        except Exception as e:
            print(f"Error using jieba tokenization: {e}")
                # Fallback to character-level tokenization
            return list(text)
    else:
        # Use fugashi tokenizer for Japanese
        if HAVE_FUGASHI:
            try:
                import fugashi
                tagger = fugashi.Tagger()
                return [word.surface for word in tagger(text)]
            except Exception as e:
                print(f"Error using fugashi tokenization: {e}")
                    # Fallback to character-level tokenization
        return list(text)  # Character-level tokenization as fallback
