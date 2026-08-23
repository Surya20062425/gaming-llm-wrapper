#!/usr/bin/env python
"""
Test script for GameMaster AI implementation.
Verifies that all modules can be imported and basic functionality works.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.model_manager import ModelManager, create_model_manager
        print("[OK] model_manager imported successfully")
    except Exception as e:
        print(f"[FAIL] model_manager import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.prompts import (
            GamingContext, PromptType, GameGenre, get_prompt,
            get_available_prompt_types, get_supported_games, detect_game_genre, BASE_SYSTEM_PROMPT
        )
        print("[OK] prompts imported successfully")
    except Exception as e:
        print(f"[FAIL] prompts import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.gaming_llm import GamingLLM, GamingResponse, create_gaming_llm
        print("[OK] gaming_llm imported successfully")
    except Exception as e:
        print(f"[FAIL] gaming_llm import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.knowledge_base import GamingKnowledgeBase, GamingDocument, create_knowledge_base, create_sample_knowledge_base
        print("[OK] knowledge_base imported successfully")
    except Exception as e:
        print(f"[FAIL] knowledge_base import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from src.cli import main, interactive_mode, single_question_mode
        print("[OK] cli imported successfully")
    except Exception as e:
        print(f"[FAIL] cli import failed: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_prompts():
    """Test prompt system."""
    print("\nTesting prompt system...")
    
    try:
        from src.prompts import (
            get_supported_games, get_available_prompt_types, 
            detect_game_genre, get_prompt, PromptType, GamingContext
        )
        
        # Test supported games
        games = get_supported_games()
        assert len(games) > 0, "No supported games"
        print(f"[OK] Supported games: {len(games)} games")
        
        # Test prompt types
        types = get_available_prompt_types()
        assert len(types) > 0, "No prompt types"
        print(f"[OK] Prompt types: {len(types)} types")
        
        # Test genre detection
        genre = detect_game_genre("League of Legends")
        assert genre.value == "moba", f"Expected MOBA, got {genre.value}"
        print(f"[OK] Genre detection works: League of Legends -> {genre.value}")
        
        # Test prompt generation
        context = GamingContext(
            game="League of Legends",
            genre=genre,
            prompt_type=PromptType.BUILD_OPTIMIZER,
            champion_hero="Jinx",
            role="ADC",
            player_rank="Diamond 2"
        )
        prompt = get_prompt(PromptType.BUILD_OPTIMIZER, context)
        assert "Jinx" in prompt, "Champion not in prompt"
        assert "ADC" in prompt, "Role not in prompt"
        print("[OK] Prompt generation works")
        
        return True
    except Exception as e:
        print(f"[FAIL] Prompt test failed: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        config_path = Path(__file__).parent / "config" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "model" in config, "Missing model config"
        assert "knowledge_base" in config, "Missing knowledge_base config"
        assert "gaming_features" in config, "Missing gaming_features config"
        print("[OK] Configuration loaded successfully")
        print(f"   Model: {config['model']['name']}")
        print(f"   Quantization: {config['model']['quantization']}")
        print(f"   Knowledge base enabled: {config['knowledge_base']['enabled']}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Config test failed: {e}")
        traceback.print_exc()
        return False


def test_gaming_llm_creation():
    """Test GamingLLM creation (without loading model)."""
    print("\nTesting GamingLLM creation...")
    
    try:
        from src.gaming_llm import create_gaming_llm
        
        config_path = Path(__file__).parent / "config" / "config.yaml"
        gaming_llm = create_gaming_llm(config_path=str(config_path))
        assert gaming_llm is not None, "Failed to create GamingLLM"
        print("[OK] GamingLLM created successfully")
        
        # Test model info (before loading)
        info = gaming_llm.get_model_info()
        assert info.get("status") == "not_loaded", f"Expected not_loaded status, got {info}"
        print(f"[OK] Model info (not loaded): {info['status']}")
        
        # Test supported games
        from src.prompts import get_supported_games
        games = get_supported_games()
        print(f"[OK] Supported games accessible: {len(games)}")
        
        return True
    except Exception as e:
        print(f"[FAIL] GamingLLM creation test failed: {e}")
        traceback.print_exc()
        return False


def test_knowledge_base():
    """Test knowledge base (without RAG dependencies)."""
    print("\nTesting knowledge base...")
    
    try:
        from src.knowledge_base import GamingKnowledgeBase, create_knowledge_base
        import yaml
        
        config_path = Path(__file__).parent / "config" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        kb = create_knowledge_base(config)
        assert kb is not None, "Failed to create knowledge base"
        print("[OK] Knowledge base created successfully")
        
        # Test stats
        stats = kb.get_stats()
        assert "enabled" in stats, "Missing enabled in stats"
        print(f"[OK] Knowledge base stats: enabled={stats['enabled']}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Knowledge base test failed: {e}")
        traceback.print_exc()
        return False


def test_cli_help():
    """Test CLI help output."""
    print("\nTesting CLI...")
    
    try:
        from src.cli import main
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        # Test --list-games
        sys.argv = ["cli", "--list-games"]
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                main()
            except SystemExit:
                pass
        output = f.getvalue()
        assert "League of Legends" in output, "Games not listed"
        print("[OK] CLI --list-games works")
        
        # Test --list-types
        sys.argv = ["cli", "--list-types"]
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                main()
            except SystemExit:
                pass
        output = f.getvalue()
        assert "Build Optimizer" in output, "Types not listed"
        print("[OK] CLI --list-types works")
        
        return True
    except Exception as e:
        print(f"[FAIL] CLI test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("GameMaster AI - Implementation Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_prompts,
        test_config_loading,
        test_gaming_llm_creation,
        test_knowledge_base,
        test_cli_help,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n[SUCCESS] All tests passed! GameMaster AI is ready to use.")
        print("\nNext steps:")
        print("  1. Run: streamlit run ui/streamlit_app.py")
        print("  2. Or: python ui/gradio_app.py")
        print("  3. Or: python -m src.cli --interactive")
        return 0
    else:
        print(f"\n[FAIL] {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())