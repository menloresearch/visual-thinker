import argparse
import logging
from evaluator import MazeBenchEvaluator
from models.vllm_model import VLLMModel
from models.hf_model import HuggingFaceModel

def parse_args():
    parser = argparse.ArgumentParser(description='MazeBench Evaluation Framework')
    
    # Model configuration
    parser.add_argument('--engine-type', choices=['vllm', 'hf'], required=True,
                       help='Model backend to use (vllm or hf)')
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
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
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
    else:  # HuggingFace
        model = HuggingFaceModel(
            model_name=args.model_name,
            prompt_template=args.instruction_type,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens
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