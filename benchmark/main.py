import argparse
import logging
from evaluator import MazeBenchEvaluator
from models.vllm_model import VLLMModel
from models.hf_model import HuggingFaceModel
from models.openai_api import OpenAIModel
import os

def parse_args():
    parser = argparse.ArgumentParser(description='MazeBench Evaluation Framework')
    
    # Model configuration
    parser.add_argument('--engine-type', choices=['vllm', 'hf', 'openai'], required=True,
                       help='Model backend to use (vllm, hf, or openai)')
    parser.add_argument('--model-name', required=True,
                       help='Name or path of the model to evaluate')
    parser.add_argument('--instruction-type', required=True,
                       help='Instruction type used to guide the model or processing task.')
    
    # Inference settings
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum number of tokens to generate')
    
    # VLLM specific settings
    parser.add_argument('--tensor-parallel-size', type=int, 
                       help='Number of GPUs for tensor parallelism (VLLM only)')
    
    # OpenAI API settings
    parser.add_argument('--api-key', type=str,
                       help='API key for OpenAI-compatible APIs')
    parser.add_argument('--api-base', type=str, default="https://api.openai.com/v1",
                       help='Base URL for API requests (OpenAI-compatible APIs)')
    parser.add_argument('--request-timeout', type=int, default=60,
                       help='Timeout for API requests in seconds (OpenAI-compatible APIs)')
    parser.add_argument('--retry-attempts', type=int, default=3,
                       help='Number of retry attempts for failed requests (OpenAI-compatible APIs)')
    parser.add_argument('--organization', type=str,
                       help='Organization ID for API requests (OpenAI-compatible APIs)')
    parser.add_argument('--skip-auth', action='store_true',
                       help='Skip authentication for local endpoints (OpenAI-compatible APIs)')
    # Output settings
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize model based on type
    if args.engine_type == 'vllm':
        model = VLLMModel(
            model_name=args.model_name,
            prompt_template=args.instruction_type,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size
        )
    elif args.engine_type == 'hf':  
        model = HuggingFaceModel(
            model_name=args.model_name,
            prompt_template=args.instruction_type,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    elif args.engine_type == 'openai':
        # Check for API key
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key is required for OpenAI models. Provide it either as a parameter "
                "with --api-key or set the OPENAI_API_KEY environment variable."
            )
        
        model = OpenAIModel(
            model_name=args.model_name,
            batch_size=args.batch_size,
            api_key=api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            retry_attempts=args.retry_attempts,
            organization=args.organization,
            skip_auth=args.skip_auth
        )
    
    # Initialize evaluator
    evaluator = MazeBenchEvaluator(model)
    
    try:
        # Run evaluation
        results = evaluator.evaluate()
        
        # Print summary results
        print("\n=== MazeBench Evaluation Results ===")
        print(f"Model: {args.model_name}")
        print(f"Backend: {args.engine_type}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        
        print("\nLevel-wise Results:")
        for level, stats in sorted(results['level_accuracies'].items()):
            print(f"Level {level}: {stats['accuracy']:.2f}% "
                  f"({stats['correct']}/{stats['total']})")
        
        # Save detailed results
        evaluator.save_results(args.output_dir)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()