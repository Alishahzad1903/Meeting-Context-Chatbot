# #!/usr/bin/env python3
# """
# Transcription and Diarization Pipeline with WhisperX

# This script transcribes audio files using WhisperX and performs speaker diarization.
# It handles dependency installation, GPU setup, and produces JSON output.
# """

# import os
# import sys
# import subprocess
# import torch
# import gc
# import json
# import logging
# import argparse
# from pathlib import Path

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)

# def install_dependencies():
#     """Install and verify all required dependencies."""
#     logger.info("Installing dependencies...")
    
#     try:
#         # Check if running in Colab
#         try:
#             import google.colab
#             IN_COLAB = True
#         except ImportError:
#             IN_COLAB = False
        
#         # Check existing PyTorch version
#         try:
#             import torch
#             torch_version = torch.__version__
#             cuda_available = torch.cuda.is_available()
#             logger.info(f"Found PyTorch {torch_version} (CUDA available: {cuda_available})")
            
#             # If it's PyTorch 2.x, we need to downgrade
#             if torch_version.startswith('2'):
#                 logger.warning("PyTorch 2.x detected, which may cause compatibility issues.")
#                 if IN_COLAB:
#                     logger.info("In Colab environment - manual action may be required.")
#                     logger.info("Consider running: !pip uninstall -y torch torchvision torchaudio")
#                     logger.info("Then: !pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116")
#                     logger.info("After installing correct versions, restart the runtime.")
#                     prompt = input("Continue anyway? (y/n): ")
#                     if prompt.lower() != 'y':
#                         sys.exit(0)
#         except ImportError:
#             logger.info("PyTorch not found, will install compatible version.")
        
#         # Install CUDA libraries if needed and if root
#         if os.geteuid() == 0:  # Running as root
#             logger.info("Installing CUDA libraries (requires root)...")
#             try:
#                 subprocess.run("apt-get update && apt-get install -y libcudnn8", shell=True, check=False)
#             except Exception as e:
#                 logger.warning(f"Failed to install CUDA libraries: {e}")
        
#         # Install WhisperX
#         try:
#             import whisperx
#             logger.info("WhisperX already installed.")
#         except ImportError:
#             logger.info("Installing WhisperX...")
#             subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/m-bain/whisperx.git"], check=True)
        
#         # Verify installations were successful
#         try:
#             import whisperx
#             logger.info("All dependencies installed successfully!")
#         except ImportError:
#             logger.error("Failed to install WhisperX. Please install it manually.")
#             sys.exit(1)
            
#     except Exception as e:
#         logger.error(f"Error during dependency installation: {e}")
#         logger.info("You may need to install dependencies manually:")
#         logger.info("pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116")
#         logger.info("pip install git+https://github.com/m-bain/whisperx.git")

# def process_audio(audio_file, model_name="large-v2", hf_token=None, 
#                   device="cuda", compute_type="float16", batch_size=8, 
#                   language=None, output_dir="output"):
#     """
#     Process an audio file with WhisperX for transcription and diarization.
    
#     Args:
#         audio_file: Path to the audio file to process
#         model_name: WhisperX model to use
#         hf_token: Hugging Face token for speaker diarization
#         device: Device to use (cuda/cpu)
#         compute_type: Computation precision (float16/float32)
#         batch_size: Batch size for processing
#         language: Language code (auto-detect if None)
#         output_dir: Directory to save the output
        
#     Returns:
#         Dictionary containing the transcription results
#     """
#     import whisperx
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Set HF token if provided
#     if hf_token:
#         os.environ["HF_TOKEN"] = hf_token
    
#     # Check device
#     if device == "cuda" and not torch.cuda.is_available():
#         logger.warning("CUDA requested but not available, falling back to CPU")
#         device = "cpu"
    
#     # Setup GPU if using CUDA
#     if device == "cuda":
#         logger.info("Setting up CUDA environment...")
#         torch.cuda.empty_cache()
#         gc.collect()
        
#         # Enable TF32 precision
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
    
#     # Step 1: Load model and transcribe
#     logger.info(f"Loading WhisperX model: {model_name}")
#     try:
#         model = whisperx.load_model(model_name, device, compute_type=compute_type)
#     except RuntimeError as e:
#         if "float16" in str(e):
#             logger.warning("float16 not supported, falling back to float32")
#             compute_type = "float32"
#             model = whisperx.load_model(model_name, device, compute_type=compute_type)
#         else:
#             raise e
    
#     logger.info(f"Transcribing audio: {audio_file}")
#     audio = whisperx.load_audio(audio_file)
#     result = model.transcribe(audio, batch_size=batch_size, language=language)
    
#     # Free up memory
#     del model
#     if device == "cuda":
#         torch.cuda.empty_cache()
#         gc.collect()
    
#     # Step 2: Perform word-level alignment
#     logger.info("Performing word-level alignment")
#     try:
#         model_a, metadata = whisperx.load_align_model(
#             language_code=result["language"],
#             device=device
#         )
#         result = whisperx.align(
#             result["segments"],
#             model_a,
#             metadata,
#             audio,
#             device,
#             return_char_alignments=False
#         )
        
#         # Free up memory
#         del model_a
#         if device == "cuda":
#             torch.cuda.empty_cache()
#             gc.collect()
#     except Exception as e:
#         logger.error(f"Alignment failed: {str(e)}")
#         logger.info("Continuing with unaligned transcription")
    
#     # Step 3: Speaker diarization if token provided
#     if hf_token:
#         logger.info("Performing speaker diarization")
#         try:
#             diarize_model = whisperx.DiarizationPipeline(
#                 use_auth_token=hf_token,
#                 device=device
#             )
#             if 
#             diarize_segments = diarize_model(audio_file)
#             result = whisperx.assign_word_speakers(diarize_segments, result)
            
#             # Free up memory
#             del diarize_model
#             if device == "cuda":
#                 torch.cuda.empty_cache()
#                 gc.collect()
#         except Exception as e:
#             logger.error(f"Speaker diarization failed: {str(e)}")
#             logger.info("Continuing without speaker diarization")
#     else:
#         logger.info("Skipping speaker diarization (no Hugging Face token provided)")
    
#     # Step 4: Save results to JSON
#     base_filename = os.path.splitext(os.path.basename(audio_file))[0]
#     output_file = os.path.join(output_dir, f"{base_filename}.json")
    
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2)
    
#     logger.info(f"Results saved to {output_file}")
    
#     return result

# def parse_arguments():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Audio Transcription and Diarization with WhisperX",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
    
#     parser.add_argument("--audio_file", type=str, required=True,
#                         help="Path to the audio file for transcription")
#     parser.add_argument("--model_name", type=str, default="large-v2",
#                         help="WhisperX model name to use")
#     parser.add_argument("--hf_token", type=str,
#                         help="Hugging Face token for speaker diarization models")
#     parser.add_argument("--device", type=str, 
#                         default="cuda" if torch.cuda.is_available() else "cpu",
#                         help="Device to use for computation (cuda/cpu)")
#     parser.add_argument("--compute_type", type=str, default="float16",
#                         help="Compute type for model inference (float16/float32)")
#     parser.add_argument("--batch_size", type=int, default=8,
#                         help="Batch size for processing")
#     parser.add_argument("--language", type=str, default=None,
#                         help="Language code (e.g., 'en'). Auto-detects if not specified.")
#     parser.add_argument("--output_dir", type=str, default="output",
#                         help="Directory to save output files")
#     parser.add_argument("--install_deps", action="store_true",
#                         help="Install dependencies before running")
    
#     return parser.parse_args()

# def print_summary(result):
#     """Print a summary of the transcription results."""
#     print("\n=== Transcription Summary ===\n")
    
#     # Print language
#     print(f"Detected language: {result.get('language', 'unknown')}")
    
#     # Print number of segments
#     segments = result.get("segments", [])
#     print(f"Number of segments: {len(segments)}")
    
#     # Print total duration
#     if segments:
#         total_duration = segments[-1]["end"]
#         print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
#     # Print speakers if available
#     unique_speakers = set()
#     for segment in segments:
#         if "speaker" in segment:
#             unique_speakers.add(segment["speaker"])
    
#     if unique_speakers:
#         print(f"Detected speakers: {', '.join(sorted(unique_speakers))}")
    
#     # Print sample of transcription
#     print("\n=== Sample Transcription ===\n")
#     for i, segment in enumerate(segments[:5], 1):
#         speaker_tag = f"[{segment.get('speaker', 'UNKNOWN')}] " if 'speaker' in segment else ""
#         print(f"{i}. [{segment['start']:.2f} - {segment['end']:.2f}] {speaker_tag}{segment['text']}")
    
#     if len(segments) > 5:
#         print(f"... and {len(segments) - 5} more segments")

# def main():
#     """Main function to run the transcription and diarization pipeline."""
#     # Parse arguments
#     args = parse_arguments()
    
#     # Check if audio file exists
#     if not os.path.isfile(args.audio_file):
#         logger.error(f"Audio file not found: {args.audio_file}")
#         sys.exit(1)
    
#     # Install dependencies if requested
#     if args.install_deps:
#         install_dependencies()
    
#     # Try to import whisperx
#     try:
#         import whisperx
#     except ImportError:
#         logger.error("WhisperX not installed. Run with --install_deps to install dependencies.")
#         sys.exit(1)
    
#     # Process the audio file
#     try:
#         result = process_audio(
#             audio_file=args.audio_file,
#             model_name=args.model_name,
#             hf_token=args.hf_token,
#             device=args.device,
#             compute_type=args.compute_type,
#             batch_size=args.batch_size,
#             language=args.language,
#             output_dir=args.output_dir
#         )
        
#         # Print summary of results
#         print_summary(result)
        
#         logger.info("Processing completed successfully!")
#     except Exception as e:
#         logger.error(f"Error during processing: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Transcription and Diarization Pipeline with WhisperX

This script transcribes audio files using WhisperX and performs speaker diarization.
It handles dependency installation, GPU setup, and produces JSON output.
"""

import os
import sys
import subprocess
import torch
import gc
import json
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install and verify all required dependencies."""
    logger.info("Installing dependencies...")
    
    try:
        # Check if running in Colab
        try:
            import google.colab
            IN_COLAB = True
        except ImportError:
            IN_COLAB = False
        
        # Check existing PyTorch version
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            logger.info(f"Found PyTorch {torch_version} (CUDA available: {cuda_available})")
            
            # If it's PyTorch 2.x, we need to downgrade
            if torch_version.startswith('2'):
                logger.warning("PyTorch 2.x detected, which may cause compatibility issues.")
                if IN_COLAB:
                    logger.info("In Colab environment - manual action may be required.")
                    logger.info("Consider running: !pip uninstall -y torch torchvision torchaudio")
                    logger.info("Then: !pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116")
                    logger.info("After installing correct versions, restart the runtime.")
                    prompt = input("Continue anyway? (y/n): ")
                    if prompt.lower() != 'y':
                        sys.exit(0)
        except ImportError:
            logger.info("PyTorch not found, will install compatible version.")
        
        # Install CUDA libraries if needed and if root
        if os.geteuid() == 0:  # Running as root
            logger.info("Installing CUDA libraries (requires root)...")
            try:
                subprocess.run("apt-get update && apt-get install -y libcudnn8", shell=True, check=False)
            except Exception as e:
                logger.warning(f"Failed to install CUDA libraries: {e}")
        
        # Install WhisperX
        try:
            import whisperx
            logger.info("WhisperX already installed.")
        except ImportError:
            logger.info("Installing WhisperX...")
            subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/m-bain/whisperx.git"], check=True)
        
        # Verify installations were successful
        try:
            import whisperx
            logger.info("All dependencies installed successfully!")
        except ImportError:
            logger.error("Failed to install WhisperX. Please install it manually.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during dependency installation: {e}")
        logger.info("You may need to install dependencies manually:")
        logger.info("pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116")
        logger.info("pip install git+https://github.com/m-bain/whisperx.git")

def process_audio(audio_file, model_name="large-v2", hf_token=None, 
                  device="cuda", compute_type="float16", batch_size=8, 
                  language="en", output_dir="output", 
                  min_speakers=None, max_speakers=None):
    """
    Process an audio file with WhisperX for transcription and diarization.
    
    Args:
        audio_file: Path to the audio file to process
        model_name: WhisperX model to use
        hf_token: Hugging Face token for speaker diarization
        device: Device to use (cuda/cpu)
        compute_type: Computation precision (float16/float32)
        batch_size: Batch size for processing
        language: Language code (auto-detect if None)
        output_dir: Directory to save the output
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        
    Returns:
        Dictionary containing the transcription results
    """
    import whisperx
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set HF token if provided
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # Setup GPU if using CUDA
    if device == "cuda":
        logger.info("Setting up CUDA environment...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Enable TF32 precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Step 1: Load model and transcribe
    logger.info(f"Loading WhisperX model: {model_name}")
    try:
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
    except RuntimeError as e:
        if "float16" in str(e):
            logger.warning("float16 not supported, falling back to float32")
            compute_type = "float32"
            model = whisperx.load_model(model_name, device, compute_type=compute_type)
        else:
            raise e
    
    logger.info(f"Transcribing audio: {audio_file}")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="en")
    
    # Free up memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Step 2: Perform word-level alignment
    logger.info("Performing word-level alignment")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
        
        # Free up memory
        del model_a
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        logger.error(f"Alignment failed: {str(e)}")
        logger.info("Continuing with unaligned transcription")
    
    # Step 3: Speaker diarization if token provided
    if hf_token:
        logger.info("Performing speaker diarization")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )
            
            # Prepare diarization kwargs
            diarize_kwargs = {}
            if min_speakers is not None:
                diarize_kwargs['min_speakers'] = min_speakers
            if max_speakers is not None:
                diarize_kwargs['max_speakers'] = max_speakers
            
            # Perform diarization with optional speaker count
            if diarize_kwargs:
                logger.info(f"Diarization parameters: {diarize_kwargs}")
                diarize_segments = diarize_model(audio_file, **diarize_kwargs)
            else:
                diarize_segments = diarize_model(audio_file)
            
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Free up memory
            del diarize_model
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            logger.error(f"Speaker diarization failed: {str(e)}")
            logger.info("Continuing without speaker diarization")
    else:
        logger.info("Skipping speaker diarization (no Hugging Face token provided)")
    
    # Step 4: Save results to JSON
    base_filename = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = os.path.join(output_dir, f"{base_filename}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Audio Transcription and Diarization with WhisperX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--audio_file", type=str, required=True,
                        help="Path to the audio file for transcription")
    parser.add_argument("--model_name", type=str, default="large-v2",
                        help="WhisperX model name to use")
    parser.add_argument("--hf_token", type=str,
                        help="Hugging Face token for speaker diarization models")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation (cuda/cpu)")
    parser.add_argument("--compute_type", type=str, default="float16",
                        help="Compute type for model inference (float16/float32)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--language", type=str, default=None,
                        help="Language code (e.g., 'en'). Auto-detects if not specified.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output files")
    parser.add_argument("--install_deps", action="store_true",
                        help="Install dependencies before running")
    parser.add_argument("--min_speakers", type=int, 
                        help="Minimum number of speakers for diarization")
    parser.add_argument("--max_speakers", type=int, 
                        help="Maximum number of speakers for diarization")
    
    return parser.parse_args()

def print_summary(result):
    """Print a summary of the transcription results."""
    print("\n=== Transcription Summary ===\n")
    
    # Print language
    print(f"Detected language: {result.get('language', 'unknown')}")
    
    # Print number of segments
    segments = result.get("segments", [])
    print(f"Number of segments: {len(segments)}")
    
    # Print total duration
    if segments:
        total_duration = segments[-1]["end"]
        print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    # Print speakers if available
    unique_speakers = set()
    for segment in segments:
        if "speaker" in segment:
            unique_speakers.add(segment["speaker"])
    
    if unique_speakers:
        print(f"Detected speakers: {', '.join(sorted(unique_speakers))}")
    
    # Print sample of transcription
    print("\n=== Sample Transcription ===\n")
    for i, segment in enumerate(segments[:5], 1):
        speaker_tag = f"[{segment.get('speaker', 'UNKNOWN')}] " if 'speaker' in segment else ""
        print(f"{i}. [{segment['start']:.2f} - {segment['end']:.2f}] {speaker_tag}{segment['text']}")
    
    if len(segments) > 5:
        print(f"... and {len(segments) - 5} more segments")

def main():
    """Main function to run the transcription and diarization pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Check if audio file exists
    if not os.path.isfile(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        install_dependencies()
    
    # Try to import whisperx
    try:
        import whisperx
    except ImportError:
        logger.error("WhisperX not installed. Run with --install_deps to install dependencies.")
        sys.exit(1)
    
    # Process the audio file
    try:
        process_kwargs = {
            'audio_file': args.audio_file,
            'model_name': args.model_name,
            'hf_token': args.hf_token,
            'device': args.device,
            'compute_type': args.compute_type,
            'batch_size': args.batch_size,
            'language': args.language,
            'output_dir': args.output_dir
        }
        
        # Conditionally add min and max speakers if specified
        if args.min_speakers is not None:
            process_kwargs['min_speakers'] = args.min_speakers
        if args.max_speakers is not None:
            process_kwargs['max_speakers'] = args.max_speakers
        
        result = process_audio(**process_kwargs)
        
        # Print summary of results
        print_summary(result)
        
        logger.info("Processing completed successfully!")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
