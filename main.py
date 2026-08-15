import argparse
import sys
import os

sys.path.append(os.path.abspath("."))

from src.pipeline import BenchmarkPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepWatermark Benchmark CLI")
    
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"⚠️ [Main] Warning: Unknown arguments ignored: {unknown}")

    print(f"🚀 [CLI] Initializing Benchmark Pipeline...")
    print(f"📂 [CLI] Config Path: {args.config}")
    
    try:
        
        pipe = BenchmarkPipeline(args)
        pipe.run()
        print("✅ [CLI] Benchmark Finished Successfully.")
        
    except Exception as e:
        print(f"❌ [CLI] Critical Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)