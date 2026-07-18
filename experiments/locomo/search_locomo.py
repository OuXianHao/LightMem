import json
from tqdm import tqdm
import datetime
import time
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import argparse

from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

from prompts import ANSWER_PROMPT, ANSWER_PROMPT_StructMem
from retrievers import QdrantEntryLoader, VectorRetriever, format_related_memories
from llm_judge import evaluate_llm_judge


# ============ Configuration ============
LOGS_ROOT = "./logs"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, f"lightmem_locomo_{RUN_TIMESTAMP}")
os.makedirs(RUN_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RUN_LOG_DIR, 'lightmem_locomo_evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vector_baseline")

# Default paths (can be overridden by command line arguments)
DEFAULT_DATA_PATH = '/path/to/locomo_dataset.json'
DEFAULT_QDRANT_DIR = './qdrant_pre_update' 
DEFAULT_EMBEDDING_MODEL_PATH = '/path/to/embedding-model'
DEFAULT_RESULTS_DIR = './lightmem_locomo_results'
DEFAULT_RETRIEVAL_LIMIT = 60


class AnswerGenerator:
    """OpenAI-compatible Qwen3 answer generation adapter."""

    def __init__(self, args):
        if args.llm_backend != "openai":
            raise ValueError("Only --llm-backend openai is supported for Qwen3 API inference")
        if not args.base_url:
            raise ValueError("--base-url or QWEN3_BASE_URL is required for Qwen3 API inference")
        if not args.api_key:
            raise ValueError("--api-key or QWEN3_API_KEY is required for Qwen3 API inference")
        from openai import OpenAI
        self.client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        self.model = args.llm_model
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_tokens = args.max_tokens
        self.base_url = args.base_url
        logger.info("[LLM Backend]\nmodel_name = %s\nbase_url = %s", self.model, self.base_url)

    def generate(self, messages):
        logger.info("[LLM Backend]\nmodel_name = %s\nbase_url = %s", self.model, self.base_url)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        usage = {
            'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
            'completion_tokens': getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
            'total_tokens': getattr(response.usage, 'total_tokens', 0) if response.usage else 0,
        }
        if not response.choices:
            raise RuntimeError("Qwen3 API returned no choices")
        return response.choices[0].message.content or "", usage


# ============ Dataset Parsing ============

def parse_locomo_dataset(data_path: str) -> List[Dict]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = {
            'sample_id': item['sample_id'],
            'conversation': item['conversation'],
            'qa': []
        }

        for qa_item in item.get('qa', []):
            answer = qa_item.get('answer') or qa_item.get('adversarial_answer', '')
            sample['qa'].append({
                'question': qa_item['question'],
                'answer': answer,
                'category': qa_item['category']
            })

        samples.append(sample)

    return samples


# ============ Retrieval Strategies ============

def retrieve_by_speaker(
    entries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    limit_per_speaker: int
) -> List[Dict]:
    """
    Retrieve top-k memories from each speaker separately.
    
    This strategy ensures balanced representation of both speakers by retrieving
    the same number of memories from each.
    
    Args:
        entries: All available memory entries
        retriever: VectorRetriever instance
        question: Query string
        limit_per_speaker: Number of memories to retrieve per speaker
        
    Returns:
        Combined list of retrieved memories with speaker annotations
    """
    # Group entries by speaker
    speaker_groups = {}
    for entry in entries:
        payload = entry.get('payload', {})
        speaker_name = payload.get('speaker_name', 'Unknown')
        
        if speaker_name not in speaker_groups:
            speaker_groups[speaker_name] = []
        speaker_groups[speaker_name].append(entry)
    
    logger.info(f"Found {len(speaker_groups)} speakers: {list(speaker_groups.keys())}")
    
    # Retrieve from each speaker separately
    all_retrieved = []
    for speaker_name, group_entries in speaker_groups.items():
        logger.info(f"Retrieving top-{limit_per_speaker} for {speaker_name}...")
        
        speaker_retrieved = retriever.retrieve(
            group_entries, 
            question, 
            limit=limit_per_speaker
        )
        
        logger.info(f"  Retrieved {len(speaker_retrieved)}/{len(group_entries)} entries")
        
        # Annotate with speaker name
        for entry in speaker_retrieved:
            entry['_retrieved_speaker'] = speaker_name
        
        all_retrieved.extend(speaker_retrieved)
    
    return all_retrieved


def retrieve_combined(
    entries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    total_limit: int
) -> List[Dict]:
    """
    Retrieve top-k memories across all speakers.
    
    This strategy retrieves memories based purely on similarity scores,
    without balancing between speakers.
    
    Args:
        entries: All available memory entries
        retriever: VectorRetriever instance
        question: Query string
        total_limit: Total number of memories to retrieve
        
    Returns:
        List of retrieved memories with speaker annotations
    """
    logger.info(f"Retrieving combined top-{total_limit} entries across speakers...")
    
    combined = retriever.retrieve(entries, question, limit=total_limit)
    
    # Annotate with speaker name
    for entry in combined:
        payload = entry.get('payload', {})
        entry['_retrieved_speaker'] = payload.get('speaker_name', 'Unknown')
    
    logger.info(f"  Combined retrieval returned {len(combined)} entries")
    return combined


def retrieve_summaries(
    summaries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    limit: int
) -> List[Dict]:
    """
    Retrieve top-k summaries.
    
    Args:
        summaries: Pre-loaded summary entries
        retriever: VectorRetriever instance
        question: Query string
        limit: Number of summaries to retrieve
        
    Returns:
        List of retrieved summaries
    """
    if not summaries:
        logger.debug("No summaries available")
        return []
    
    logger.debug(f"Retrieving top-{limit} from {len(summaries)} summaries")
    retrieved = retriever.retrieve(summaries, question, limit=limit)
    
    return retrieved


# ============ Prompt Construction ============

def format_summaries(summaries: List[Dict]) -> str:
    """Format summaries for inclusion in prompt."""
    if not summaries:
        return "No session summaries available."
    
    lines = []
    for summary in summaries:
        payload = summary.get('payload', {})
        summary_text = payload.get('summary', payload.get('memory', ''))
        lines.append(f"{summary_text}")
    
    return "\n".join(lines)


def build_prompt_with_speaker_memories(
    question: str,
    retrieved_entries: List[Dict],
    enable_summary: bool = False,
    summaries: Optional[List[Dict]] = None
) -> str:
    """
    Build prompt with memories organized by speaker.
    
    Args:
        question: The question to answer
        retrieved_entries: Retrieved memory entries with speaker annotations
        enable_summary: Whether to include summaries in the prompt
        summaries: Retrieved summary entries (only used if enable_summary=True)
        
    Returns:
        Formatted prompt string ready for LLM
    """
    # Group by speaker
    speaker_groups = {}
    for entry in retrieved_entries:
        speaker_name = entry.get('_retrieved_speaker', 
                                 entry.get('payload', {}).get('speaker_name', 'Unknown'))
        if speaker_name not in speaker_groups:
            speaker_groups[speaker_name] = []
        speaker_groups[speaker_name].append(entry)
    
    speaker_names = list(speaker_groups.keys())
    
    # Handle different speaker configurations
    if len(speaker_names) == 0:
        speaker_1_name = "Speaker 1"
        speaker_2_name = "Speaker 2"
        speaker_1_memories = "No memories available."
        speaker_2_memories = "No memories available."
    elif len(speaker_names) == 1:
        speaker_1_name = speaker_names[0]
        speaker_2_name = "Speaker 2"
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = "No memories available."
    else:
        speaker_1_name = speaker_names[0]
        speaker_2_name = speaker_names[1]
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = format_related_memories(speaker_groups[speaker_2_name])
        
        logger.debug(
            f"Formatted memories - {speaker_1_name}: {len(speaker_groups[speaker_1_name])}, "
            f"{speaker_2_name}: {len(speaker_groups[speaker_2_name])}"
        )
    
    # Choose prompt template based on whether summaries are enabled
    if enable_summary:
        # Format summaries
        session_summaries = format_summaries(summaries) if summaries else "No session summaries available."
        
        # Fill StructMem prompt template
        prompt = ANSWER_PROMPT_StructMem.format(
            speaker_1_name=speaker_1_name,
            speaker_1_memories=speaker_1_memories,
            speaker_2_name=speaker_2_name,
            speaker_2_memories=speaker_2_memories,
            session_summaries=session_summaries,
            question=question
        )
    else:
        # Fill standard prompt template (no summaries)
        prompt = ANSWER_PROMPT.format(
            speaker_1_name=speaker_1_name,
            speaker_1_memories=speaker_1_memories,
            speaker_2_name=speaker_2_name,
            speaker_2_memories=speaker_2_memories,
            question=question
        )
    
    return prompt


# ============ Sample Processing ============

def process_sample(
    sample: Dict,
    entry_loader: QdrantEntryLoader,
    retriever: VectorRetriever,
    answer_generator: AnswerGenerator,
    judge_client: Any,
    llm_model: str,
    judge_model: str,
    allow_categories: List[int],
    limit_per_speaker: int,
    total_limit: int,
    retrieval_mode: str,
    enable_summary: bool = False,
    summary_limit: int = 5,
    judge_base_url: str = "configured judge client"
) -> Dict:
    """
    Process a single sample with all its QA pairs.
    
    Args:
        sample: Sample dictionary containing conversation and QA pairs
        entry_loader: QdrantEntryLoader instance
        retriever: VectorRetriever instance
        llm_client: OpenAI client for answer generation
        judge_client: OpenAI client for evaluation
        llm_model: Model name for answer generation
        judge_model: Model name for evaluation
        allow_categories: List of allowed QA categories
        limit_per_speaker: Retrieval limit per speaker (for per-speaker mode)
        total_limit: Total retrieval limit (for combined mode)
        retrieval_mode: 'per-speaker' or 'combined'
        enable_summary: Whether to retrieve and use summaries
        summary_limit: Retrieval limit for summaries
        
    Returns:
        Dictionary with sample results and statistics
    """
    sample_id = sample['sample_id']
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing sample: {sample_id}")
    logger.info(f"{'='*80}")
    
    # Initialize token statistics
    sample_token_stats = {
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'api_calls': 0
    }
    
    # Load memory entries
    try:
        entries = entry_loader.load_entries(sample_id, with_vectors=True)
        
        # Load summaries if enabled
        summaries = []
        if enable_summary:
            summaries = entry_loader.load_summaries(sample_id, with_vectors=True)
            logger.info(
                f"[{sample_id}] Loaded {len(entries)} entries + {len(summaries)} summaries"
            )
        else:
            logger.info(f"[{sample_id}] Loaded {len(entries)} entries")
        
        if not entries:
            logger.error(f"[{sample_id}] No entries loaded")
            return {
                'sample_id': sample_id,
                'error': 'No entries loaded',
                'results': [],
                'token_stats': sample_token_stats
            }
    except Exception as e:
        logger.error(f"[{sample_id}] Failed to load entries: {e}")
        return {
            'sample_id': sample_id,
            'error': str(e),
            'results': [],
            'token_stats': sample_token_stats
        }
    
    # Process each QA pair
    qa_results = []
    for qa_idx, qa in enumerate(sample['qa']):
        category = qa['category']
        
        # Skip category 5 and disallowed categories
        if int(category) == 5 or category not in allow_categories:
            continue
        
        question = qa['question']
        reference = qa['answer']
        
        logger.info(f"\n[{sample_id}] Question {qa_idx+1} (Category {category})")
        logger.info(f"Q: {question}")
        logger.info(f"A: {reference}")
        
        # Retrieve relevant memories
        time_start = time.time()
        
        # Retrieve summaries if enabled
        retrieved_summaries = []
        if enable_summary and summaries:
            retrieved_summaries = retrieve_summaries(summaries, retriever, question, summary_limit)
        
        # Retrieve entries based on mode
        if retrieval_mode == 'per-speaker':
            retrieved_entries = retrieve_by_speaker(
                entries, retriever, question, limit_per_speaker
            )
        else:
            retrieved_entries = retrieve_combined(
                entries, retriever, question, total_limit
            )
        
        retrieval_time = time.time() - time_start
        
        if not retrieved_entries:
            logger.warning(f"[{sample_id}] No entries retrieved")
            qa_results.append({
                'question': question,
                'prediction': '',
                'reference': reference,
                'category': category,
                'retrieved_count': 0,
                'summary_count': 0 if enable_summary else None,
                'retrieval_time': retrieval_time,
                'speaker_distribution': {},
                'error': 'No entries retrieved',
                'metrics': {},
                'token_usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            })
            continue
        
        # Calculate speaker distribution
        speaker_dist = {}
        for entry in retrieved_entries:
            speaker = entry.get('_retrieved_speaker', 'Unknown')
            speaker_dist[speaker] = speaker_dist.get(speaker, 0) + 1
        
        if enable_summary:
            logger.info(
                f"[{sample_id}] Retrieved {len(retrieved_summaries)} summaries + "
                f"{len(retrieved_entries)} entries in {retrieval_time:.3f}s"
            )
        else:
            logger.info(
                f"[{sample_id}] Retrieved {len(retrieved_entries)} entries in {retrieval_time:.3f}s"
            )
        logger.info(f"[{sample_id}] Speaker distribution: {speaker_dist}")
        
        # Build prompt
        user_prompt = build_prompt_with_speaker_memories(
            question, 
            retrieved_entries,
            enable_summary=enable_summary,
            summaries=retrieved_summaries if enable_summary else None
        )
        
        # Generate answer
        logger.info("[LLM Backend] query answering")
        logger.info("model_name = %s", llm_model)
        logger.info("base_url = %s", answer_generator.base_url)
        token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        try:
            generated_answer, token_usage = answer_generator.generate([
                {"role": "system", "content": "Answer the user's question using only the retrieved LightMem memories."},
                {"role": "user", "content": user_prompt},
            ])
            sample_token_stats['total_prompt_tokens'] += token_usage.get('prompt_tokens', 0)
            sample_token_stats['total_completion_tokens'] += token_usage.get('completion_tokens', 0)
            sample_token_stats['total_tokens'] += token_usage.get('total_tokens', 0)
            sample_token_stats['api_calls'] += 1
            logger.info(
                f"[{sample_id}] Token usage - Prompt: {token_usage.get('prompt_tokens', 0)}, "
                f"Completion: {token_usage.get('completion_tokens', 0)}, "
                f"Total: {token_usage.get('total_tokens', 0)}"
            )
            logger.info(f"[{sample_id}] Generated: {generated_answer}")
        except Exception as e:
            logger.error(f"[{sample_id}] Failed to generate answer: {e}")
            generated_answer = ""
        
        # Evaluate with LLM judge
        logger.info("[LLM Backend] LoCoMo judge")
        logger.info("model_name = %s", judge_model)
        logger.info("base_url = %s", judge_base_url)
        try:
            label = evaluate_llm_judge(
                question, reference, generated_answer,
                client_obj=judge_client, model_name=judge_model
            )
            metrics = {
                'judge_correct': int(label),
                'judge_response': 'CORRECT' if int(label) == 1 else 'WRONG'
            }
            logger.info(
                f"[{sample_id}] Judge: {'CORRECT' if int(label) == 1 else 'WRONG'}"
            )
        except Exception as e:
            logger.error(f"[{sample_id}] Judge evaluation failed: {e}")
            metrics = {'judge_correct': 0, 'judge_response': ''}
        
        # Store results
        result_dict = {
            'question': question,
            'prediction': generated_answer,
            'reference': reference,
            'category': category,
            'retrieved_count': len(retrieved_entries),
            'speaker_distribution': speaker_dist,
            'retrieval_time': retrieval_time,
            'metrics': metrics,
            'token_usage': token_usage
        }
        
        # Add summary count if enabled
        if enable_summary:
            result_dict['summary_count'] = len(retrieved_summaries)
        
        qa_results.append(result_dict)
    
    return {
        'sample_id': sample_id,
        'results': qa_results,
        'token_stats': sample_token_stats
    }


# ============ Main Execution ============

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Vector baseline evaluation for LoCoMo dataset"
    )
    
    # Data paths
    parser.add_argument('--dataset', '--data-file', dest='dataset', type=str, default=DEFAULT_DATA_PATH,
                       help="Path to dataset JSON file")
    parser.add_argument('--qdrant-dir', type=str, default=DEFAULT_QDRANT_DIR,
                       help="Path to Qdrant data directory")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_RESULTS_DIR,
                       help="Output directory for results")
    parser.add_argument('--output-file', type=str, default=None,
                       help="Optional final summary JSON path (in addition to per-sample files)")
    
    # Retrieval configuration
    parser.add_argument('--limit-per-speaker', type=int, default=DEFAULT_RETRIEVAL_LIMIT,
                       help="Retrieval limit per speaker (for per-speaker mode)")
    parser.add_argument('--total-limit', type=int, default=DEFAULT_RETRIEVAL_LIMIT,
                       help="Total retrieval limit (for combined mode)")
    parser.add_argument('--retrieval-mode', type=str, 
                       choices=['combined', 'per-speaker'], default='combined',
                       help="Retrieval strategy")
    parser.add_argument('--allow-categories', type=int, nargs='+', default=[1, 2, 3, 4],
                       help="Allowed QA categories")
    parser.add_argument('--embedder', type=str, 
                       choices=['huggingface', 'openai'], default='huggingface',
                       help="Embedding backend")
    parser.add_argument('--embedding-model-path', type=str, 
                       default=DEFAULT_EMBEDDING_MODEL_PATH,
                       help="Path to embedding model (for huggingface backend)")
    
    # Summary configuration
    parser.add_argument('--enable-summary', action='store_true',
                       help="Enable summary retrieval (StructMem mode)")
    parser.add_argument('--summary-limit', type=int, default=5,
                       help="Retrieval limit for summaries (only used if --enable-summary)")
    
    # LLM configuration
    parser.add_argument('--api-key', '--llm-api-key', dest='api_key', type=str, default=os.getenv('QWEN3_API_KEY'),
                       help="Qwen3 API key (or QWEN3_API_KEY)")
    parser.add_argument('--base-url', '--llm-base-url', dest='base_url', type=str, default=os.getenv('QWEN3_BASE_URL'),
                       help="OpenAI-compatible Qwen3 API base URL (or QWEN3_BASE_URL)")
    parser.add_argument('--llm-model', '--model', dest='llm_model', type=str, default='Qwen3-30B-A3B-Instruct-2507',
                       help="LLM model name")
    parser.add_argument('--judge-api-key', type=str, default=None,
                       help="API key for judge; defaults to --api-key so evaluation can also use Qwen3")
    parser.add_argument('--judge-base-url', type=str, default=None,
                       help="Base URL for judge API; defaults to --base-url")
    parser.add_argument('--judge-model', type=str, default=None,
                       help="Judge model name; defaults to --model")
    parser.add_argument('--llm-backend', choices=['openai'], default='openai',
                       help="OpenAI-compatible API backend for Qwen3")
    parser.add_argument('--temperature', type=float, default=0.0,
                       help="Answer generation temperature")
    parser.add_argument('--top-p', type=float, default=0.9,
                       help="Answer generation nucleus sampling p")
    parser.add_argument('--max-tokens', type=int, default=1024,
                       help="Maximum answer tokens to generate")
    
    args = parser.parse_args()
    if args.llm_backend != 'openai':
        raise ValueError("Only --llm-backend openai is supported for the Qwen3 LoCoMo workflow")
    if not args.api_key:
        raise ValueError("--api-key or QWEN3_API_KEY is required for Qwen3 query answering")
    if not args.base_url:
        raise ValueError("--base-url or QWEN3_BASE_URL is required for Qwen3 query answering")
    args.judge_api_key = args.judge_api_key or args.api_key
    args.judge_base_url = args.judge_base_url or args.base_url
    args.judge_model = args.judge_model or args.llm_model
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("LightMem Evaluation - LoCoMo Dataset")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Dataset:         {args.dataset}")
    logger.info(f"  Qdrant dir:      {args.qdrant_dir}")
    logger.info(f"  Output dir:      {args.output_dir}")
    logger.info(f"  Retrieval mode:  {args.retrieval_mode}")
    if args.retrieval_mode == 'per-speaker':
        logger.info(f"  Limit per speaker: {args.limit_per_speaker}")
    else:
        logger.info(f"  Total limit:     {args.total_limit}")
    logger.info(f"  Summary enabled: {args.enable_summary}")
    if args.enable_summary:
        logger.info(f"  Summary limit:   {args.summary_limit}")
    logger.info(f"  Categories:      {args.allow_categories}")
    logger.info(f"  Embedder:        {args.embedder}")
    logger.info(f"  LLM model:       {args.llm_model}")
    logger.info(f"  Judge model:     {args.judge_model}")
    logger.info("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    logger.info("\nInitializing components...")
    
    # Initialize entry loader with summary support if enabled
    if args.enable_summary:
        entry_loader = QdrantEntryLoader(args.qdrant_dir, summary_suffix="_summary")
    else:
        entry_loader = QdrantEntryLoader(args.qdrant_dir)
    
    # Initialize embedding model
    if args.embedder == 'openai':
        embedder_cfg = BaseTextEmbedderConfig(
            model='text-embedding-3-small',
            api_key=args.api_key,
            openai_base_url=args.base_url,
            embedding_dims=1536,
        )
        embedder = TextEmbedderOpenAI(embedder_cfg)
    else:
        embedder_cfg = BaseTextEmbedderConfig(
            model=args.embedding_model_path,
            embedding_dims=384,
            model_kwargs={"device": "cuda"},
        )
        embedder = TextEmbedderHuggingface(embedder_cfg)
    
    retriever = VectorRetriever(embedder)
    
    # Create answer generator and judge client. The judge remains OpenAI-compatible
    # because LoCoMo scoring uses a separate LLM judge.
    answer_generator = AnswerGenerator(args)
    from openai import OpenAI
    judge_client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    
    logger.info(f"LLM generator initialized: {args.llm_model} ({args.llm_backend})")
    logger.info(f"Judge client initialized: {args.judge_model}")
    logger.info("[LLM Backend] LoCoMo evaluation")
    logger.info("model_name = %s", args.llm_model)
    logger.info("base_url = %s", args.base_url)
    
    # Load dataset
    logger.info(f"\nLoading dataset from {args.dataset}")
    samples = parse_locomo_dataset(args.dataset)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Initialize global statistics
    global_token_stats = {
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'total_api_calls': 0
    }
    
    # Process all samples
    all_results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total_summaries_used = 0
    
    for sample in tqdm(samples, desc="Processing samples"):
        sample_result = process_sample(
            sample, entry_loader, retriever,
            answer_generator, judge_client,
            args.llm_model, args.judge_model,
            args.allow_categories, args.limit_per_speaker,
            args.total_limit, args.retrieval_mode,
            enable_summary=args.enable_summary,
            summary_limit=args.summary_limit,
            judge_base_url=args.judge_base_url
        )
        
        all_results.append(sample_result)
        
        # Update global statistics
        sample_token_stats = sample_result.get('token_stats', {})
        global_token_stats['total_prompt_tokens'] += sample_token_stats.get('total_prompt_tokens', 0)
        global_token_stats['total_completion_tokens'] += sample_token_stats.get('total_completion_tokens', 0)
        global_token_stats['total_tokens'] += sample_token_stats.get('total_tokens', 0)
        global_token_stats['total_api_calls'] += sample_token_stats.get('api_calls', 0)
        
        # Collect metrics
        for qa_result in sample_result.get('results', []):
            total_questions += 1
            category = qa_result['category']
            category_counts[category] += 1
            all_metrics.append(qa_result['metrics'])
            all_categories.append(category)
            
            # Count summaries if enabled
            if args.enable_summary and 'summary_count' in qa_result:
                total_summaries_used += qa_result['summary_count']
        
        # Save individual sample result
        sample_file = os.path.join(args.output_dir, f"sample_{sample['sample_id']}.json")
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_result, f, ensure_ascii=False, indent=2)
    
    # Calculate aggregate metrics
    logger.info("\nCalculating aggregate metrics...")
    category_scores = {}
    total_scores = []
    
    for cat, m in zip(all_categories, all_metrics):
        score = float(m.get('judge_correct', 0)) if isinstance(m, dict) else 0.0
        total_scores.append(score)
        category_scores.setdefault(int(cat), []).append(score)
    
    aggregate_results = {"overall": {}}
    if total_scores:
        aggregate_results["overall"]["judge_correct"] = {
            "mean": float(np.mean(total_scores)),
            "std": float(np.std(total_scores)),
            "count": int(len(total_scores)),
        }
    
    for cat in sorted(category_scores.keys()):
        vals = category_scores[cat]
        if vals:
            aggregate_results[f"category_{cat}"] = {
                "judge_correct": {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "count": int(len(vals)),
                }
            }
    
    # Build config dict
    config_dict = {
        "retrieval_mode": args.retrieval_mode,
        "limit_per_speaker": args.limit_per_speaker,
        "total_limit": args.total_limit,
        "embedder": args.embedder,
        "method": "structmem" if args.enable_summary else "lightmem",
        "allow_categories": args.allow_categories,
        "enable_summary": args.enable_summary,
    }
    if args.enable_summary:
        config_dict["summary_limit"] = args.summary_limit
    
    # Save summary results
    final_results = {
        "llm_model": args.llm_model,
        "judge_model": args.judge_model,
        "dataset": args.dataset,
        "total_questions": total_questions,
        "total_samples": len(samples),
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "config": config_dict,
        "aggregate_metrics": aggregate_results,
        "token_statistics": {
            "total_prompt_tokens": global_token_stats['total_prompt_tokens'],
            "total_completion_tokens": global_token_stats['total_completion_tokens'],
            "total_tokens": global_token_stats['total_tokens'],
            "total_api_calls": global_token_stats['total_api_calls'],
            "avg_prompt_tokens_per_call": (
                global_token_stats['total_prompt_tokens'] / global_token_stats['total_api_calls']
                if global_token_stats['total_api_calls'] > 0 else 0
            ),
            "avg_completion_tokens_per_call": (
                global_token_stats['total_completion_tokens'] / global_token_stats['total_api_calls']
                if global_token_stats['total_api_calls'] > 0 else 0
            ),
            "avg_total_tokens_per_call": (
                global_token_stats['total_tokens'] / global_token_stats['total_api_calls']
                if global_token_stats['total_api_calls'] > 0 else 0
            )
        },
        "timestamp": RUN_TIMESTAMP
    }
    
    # Add summary statistics if enabled
    if args.enable_summary:
        final_results["retrieval_statistics"] = {
            "total_summaries_used": total_summaries_used,
            "avg_summaries_per_question": total_summaries_used / total_questions if total_questions > 0 else 0,
        }
    
    summary_file = args.output_file or os.path.join(args.output_dir, "summary.json")
    if os.path.dirname(summary_file):
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete")
    logger.info("=" * 80)
    logger.info(f"Total samples:    {len(samples)}")
    logger.info(f"Total questions:  {total_questions}")
    logger.info(f"LLM model:        {args.llm_model}")
    logger.info(f"Judge model:      {args.judge_model}")
    
    if args.enable_summary:
        logger.info("\nRetrieval Statistics:")
        logger.info(f"  Total summaries:  {total_summaries_used}")
        if total_questions > 0:
            logger.info(f"  Avg summaries/Q:  {total_summaries_used/total_questions:.2f}")
    
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        if count > 0:
            logger.info(
                f"  Category {category}: {count} questions "
                f"({count/total_questions*100:.1f}%)"
            )
    
    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict):
                logger.info(f"  {metric_name}:")
                for stat_name, value in stats.items():
                    logger.info(f"    {stat_name}: {value:.4f}")
    
    logger.info("\nToken Statistics:")
    logger.info(f"  Total API calls:        {global_token_stats['total_api_calls']}")
    logger.info(f"  Total prompt tokens:    {global_token_stats['total_prompt_tokens']:,}")
    logger.info(f"  Total completion tokens: {global_token_stats['total_completion_tokens']:,}")
    logger.info(f"  Total tokens:           {global_token_stats['total_tokens']:,}")
    if global_token_stats['total_api_calls'] > 0:
        logger.info(
            f"  Avg prompt/call:        "
            f"{global_token_stats['total_prompt_tokens'] / global_token_stats['total_api_calls']:.2f}"
        )
        logger.info(
            f"  Avg completion/call:    "
            f"{global_token_stats['total_completion_tokens'] / global_token_stats['total_api_calls']:.2f}"
        )
        logger.info(
            f"  Avg total/call:         "
            f"{global_token_stats['total_tokens'] / global_token_stats['total_api_calls']:.2f}"
        )
    
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()