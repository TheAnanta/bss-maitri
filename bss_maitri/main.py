"""
Main entry point for BSS Maitri AI Assistant
"""

import logging
import argparse
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_ollama():
    """Setup and check Ollama installation"""
    try:
        from .models.ollama_client import OllamaClient
        
        client = OllamaClient()
        
        print("🔍 Checking Ollama model availability...")
        
        if not client.is_model_available():
            print(f"📥 Model {client.model_name} not found. Downloading...")
            print("⚠️  This may take several minutes depending on your internet connection.")
            
            if client.pull_model():
                print("✅ Model downloaded successfully!")
            else:
                print("❌ Failed to download model. Please check:")
                print("   1. Ollama is installed and running")
                print("   2. Internet connection is available")
                print("   3. Sufficient disk space")
                return False
        else:
            print("✅ Model is ready!")
            
        return True
        
    except ImportError:
        print("❌ Ollama package not found. Please install it:")
        print("   pip install ollama")
        return False
    except Exception as e:
        print(f"❌ Error setting up Ollama: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BSS Maitri - AI Assistant for Crew Well-being"
    )
    
    parser.add_argument(
        "--mode",
        choices=["web", "cli", "setup"],
        default="web",
        help="Run mode: web interface, CLI mode, or setup only"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web interface (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create shareable public link for web interface"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 BSS Maitri - AI Assistant for Crew Well-being")
    print("=" * 50)
    
    # Setup mode
    if args.mode == "setup":
        print("🔧 Running setup...")
        if setup_ollama():
            print("✅ Setup completed successfully!")
        else:
            print("❌ Setup failed. Please check the errors above.")
            sys.exit(1)
        return
    
    # Check Ollama setup
    if not setup_ollama():
        print("❌ Ollama setup failed. Run with --mode setup to diagnose issues.")
        sys.exit(1)
    
    # Web interface mode
    if args.mode == "web":
        try:
            from .ui.web_interface import MaitriWebInterface
            
            print(f"🌐 Starting web interface on port {args.port}")
            
            interface = MaitriWebInterface()
            interface.launch(share=args.share, port=args.port)
            
        except ImportError as e:
            print(f"❌ Missing dependencies for web interface: {e}")
            print("Please install required packages:")
            print("   pip install gradio")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error starting web interface: {e}")
            sys.exit(1)
    
    # CLI mode
    elif args.mode == "cli":
        try:
            from .models.ollama_client import OllamaClient
            from .utils.multimodal_analyzer import MultimodalEmotionAnalyzer
            
            print("💬 Starting CLI mode...")
            print("Type 'quit' or 'exit' to stop.")
            print("-" * 30)
            
            client = OllamaClient()
            analyzer = MultimodalEmotionAnalyzer()
            conversation_history = []
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye! Take care of yourself.")
                        break
                    
                    if not user_input:
                        continue
                    
                    conversation_history.append(f"User: {user_input}")
                    
                    response = client.provide_companionship(
                        conversation_history,
                        "neutral"  # Default emotion in CLI mode
                    )
                    
                    print(f"\nMaitri: {response}")
                    
                    conversation_history.append(f"Maitri: {response}")
                    
                    # Keep conversation history manageable
                    if len(conversation_history) > 20:
                        conversation_history = conversation_history[-10:]
                
                except KeyboardInterrupt:
                    print("\n👋 Goodbye! Take care of yourself.")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        except Exception as e:
            print(f"❌ Error in CLI mode: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()