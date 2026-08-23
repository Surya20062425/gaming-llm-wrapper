"""
Streamlit Web UI for GameMaster AI.
Beautiful web interface for the gaming LLM assistant.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any
import yaml
from pathlib import Path

# Import our gaming LLM
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.gaming_llm import GamingLLM, create_gaming_llm
from src.prompts import PromptType, get_supported_games, get_available_prompt_types, GameGenre

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="🎮 GameMaster AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .game-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .response-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd6 0%, #6a4190 100%);
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_gaming_llm(config_path: str = "config/config.yaml") -> GamingLLM:
    """Load and cache the Gaming LLM instance."""
    try:
        gaming_llm = create_gaming_llm(config_path=config_path)
        return gaming_llm
    except Exception as e:
        st.error(f"Failed to initialize GameMaster AI: {e}")
        logger.exception("Failed to load gaming LLM")
        return None


@st.cache_resource
def load_model(_gaming_llm: GamingLLM) -> bool:
    """Load the model with caching."""
    with st.spinner("🔄 Loading AI model... This may take a minute on first run."):
        return _gaming_llm.load_model()


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎮 GameMaster AI</h1>
        <p>Your Intelligent Gaming Companion - Builds, Strategies, Lore & Coaching</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(gaming_llm: GamingLLM) -> Dict[str, Any]:
    """Render sidebar and return user selections."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model status
        if gaming_llm.is_model_loaded():
            st.success("✅ Model Loaded")
            info = gaming_llm.get_model_info()
            st.caption(f"Model: {info.get('model_id', 'Unknown').split('/')[-1]}")
            st.caption(f"Device: {info.get('device', 'Unknown')}")
            st.caption(f"Quantization: {info.get('quantization', 'Unknown')}")
        else:
            st.warning("⚠️ Model Not Loaded")
            if st.button("🔄 Load Model", use_container_width=True):
                load_model(gaming_llm)
                st.rerun()
        
        st.divider()
        
        # Game selection
        st.markdown("### 🎮 Select Game")
        games = get_supported_games()
        game = st.selectbox(
            "Game",
            games,
            index=0,
            help="Choose the game you need help with"
        )
        
        # Detect genre for display
        from src.prompts import detect_game_genre
        genre = detect_game_genre(game)
        st.caption(f"Genre: {genre.value.upper()}")
        
        st.divider()
        
        # Prompt type selection
        st.markdown("### 🎯 Assistance Type")
        prompt_types = get_available_prompt_types()
        prompt_type_labels = {
            PromptType.BUILD_OPTIMIZER: "🏗️ Build Optimizer",
            PromptType.STRATEGY_ADVISOR: "🧠 Strategy Advisor",
            PromptType.LORE_EXPLORER: "📜 Lore Explorer",
            PromptType.PATCH_ANALYZER: "📋 Patch Analyzer",
            PromptType.META_ANALYZER: "📊 Meta Analyzer",
            PromptType.COACH_MODE: "🎓 Coach Mode",
            PromptType.COUNTER_PICK: "⚔️ Counter Picks",
            PromptType.TEAM_COMP: "👥 Team Composition",
            PromptType.ITEMIZATION: "🛡️ Itemization Guide",
            PromptType.MECHANICS_GUIDE: "⚡ Mechanics Guide",
            PromptType.GENERAL_GAMING: "❓ General Question",
        }
        
        prompt_type = st.selectbox(
            "What do you need?",
            prompt_types,
            format_func=lambda x: prompt_type_labels.get(x, x.value),
            index=0
        )
        
        st.divider()
        
        # Context inputs based on prompt type
        context = {"game": game, "prompt_type": prompt_type}
        
        # Common fields
        context["player_rank"] = st.text_input(
            "🏆 Your Rank (optional)",
            placeholder="e.g., Gold 2, Diamond 1, Level 50",
            help="Your current rank/skill level for tailored advice"
        ) or None
        
        context["patch_version"] = st.text_input(
            "📦 Patch Version (optional)",
            placeholder="e.g., 14.12, 7.35, Season 12",
            help="Specific patch for accurate meta/build advice"
        ) or None
        
        # Role-specific fields
        if prompt_type in [PromptType.BUILD_OPTIMIZER, PromptType.ITEMIZATION, 
                          PromptType.MECHANICS_GUIDE, PromptType.COACH_MODE,
                          PromptType.COUNTER_PICK]:
            context["role"] = st.text_input(
                "🎭 Role (optional)",
                placeholder="e.g., ADC, Mid, Top, Jungle, Support, Carry, Tank",
                help="Your role in the game"
            ) or None
        
        # Champion/hero fields
        if prompt_type in [PromptType.BUILD_OPTIMIZER, PromptType.ITEMIZATION,
                          PromptType.MECHANICS_GUIDE, PromptType.COACH_MODE]:
            context["champion_hero"] = st.text_input(
                "⚔️ Champion/Hero/Character",
                placeholder="e.g., Jinx, Pudge, Jett, Reaper",
                help="The specific character you want help with"
            ) or None
        
        if prompt_type == PromptType.COUNTER_PICK:
            context["champion_hero"] = st.text_input(
                "🎯 Enemy Champion/Hero",
                placeholder="e.g., Yasuo, Anti-Mage, Genji",
                help="The enemy character you want to counter"
            ) or None
            context["role"] = st.text_input(
                "🎭 Your Role (optional)",
                placeholder="e.g., Mid, Carry, DPS",
                help="Your role for better counter suggestions"
            ) or None
        
        if prompt_type == PromptType.COACH_MODE:
            context["struggles"] = st.text_area(
                "🎯 What are you struggling with?",
                placeholder="e.g., 'I lose lane against assassins', 'My teamfight positioning is bad', 'I can't climb from Gold'",
                help="Describe your main challenges for personalized coaching"
            ) or None
        
        if prompt_type == PromptType.TEAM_COMP:
            context["additional_context"] = st.text_area(
                "👥 Current Team Picks / Situation",
                placeholder="e.g., 'We have Malphite top, Elise jungle, need mid and bot'",
                help="Describe current picks or what you're looking for"
            ) or None
        
        if prompt_type == PromptType.STRATEGY_ADVISOR:
            context["additional_context"] = st.text_area(
                "🧠 Situation Description",
                placeholder="e.g., 'How to play from behind as ADC', 'Early game aggression vs scaling'",
                help="Describe the strategic situation you need help with"
            ) or None
        
        if prompt_type == PromptType.LORE_EXPLORER:
            context["champion_hero"] = st.text_input(
                "👤 Character/Faction/Location (optional)",
                placeholder="e.g., Jinx, The Void, Runeterra, Shadow Isles",
                help="Specific lore element to explore"
            ) or None
            context["additional_context"] = st.text_area(
                "📜 Lore Topic",
                placeholder="e.g., 'Origin of the Void', 'Relationship between Jinx and Vi', 'History of Noxus'",
                help="What lore aspect interests you?"
            ) or None
        
        if prompt_type == PromptType.PATCH_ANALYZER:
            context["additional_context"] = st.text_area(
                "🔍 Focus Area (optional)",
                placeholder="e.g., 'Jungle changes', 'Item reworks', 'Specific champion buffs'",
                help="What aspect of the patch to focus on"
            ) or None
        
        if prompt_type == PromptType.META_ANALYZER:
            context["role"] = st.text_input(
                "🎭 Role Focus (optional)",
                placeholder="e.g., Jungle, Support, All roles",
                help="Focus on specific role meta"
            ) or None
        
        if prompt_type == PromptType.GENERAL_GAMING:
            context["additional_context"] = st.text_area(
                "❓ Your Question",
                placeholder="Ask anything about the game...",
                help="General gaming question"
            ) or None
        
        st.divider()
        
        # Generation parameters
        with st.expander("🔧 Advanced Settings"):
            context["temperature"] = st.slider(
                "Temperature",
                0.1, 1.5, 0.7, 0.1,
                help="Higher = more creative, Lower = more focused"
            )
            context["max_new_tokens"] = st.slider(
                "Max Response Length",
                512, 4096, 2048, 256,
                help="Maximum tokens in response"
            )
            context["top_p"] = st.slider(
                "Top-p",
                0.1, 1.0, 0.9, 0.05,
                help="Nucleus sampling threshold"
            )
        
        # Clear history button
        if st.button("🗑️ Clear Conversation History", use_container_width=True):
            gaming_llm.clear_history()
            st.success("History cleared!")
            st.rerun()
        
        return context


def render_main_content(gaming_llm: GamingLLM, context: Dict[str, Any]):
    """Render main content area."""
    prompt_type = context["prompt_type"]
    game = context["game"]
    
    # Title based on prompt type
    prompt_titles = {
        PromptType.BUILD_OPTIMIZER: "🏗️ Build Optimizer",
        PromptType.STRATEGY_ADVISOR: "🧠 Strategy Advisor",
        PromptType.LORE_EXPLORER: "📜 Lore Explorer",
        PromptType.PATCH_ANALYZER: "📋 Patch Analyzer",
        PromptType.META_ANALYZER: "📊 Meta Analyzer",
        PromptType.COACH_MODE: "🎓 Coach Mode",
        PromptType.COUNTER_PICK: "⚔️ Counter Picks",
        PromptType.TEAM_COMP: "👥 Team Composition",
        PromptType.ITEMIZATION: "🛡️ Itemization Guide",
        PromptType.MECHANICS_GUIDE: "⚡ Mechanics Guide",
        PromptType.GENERAL_GAMING: "❓ General Gaming",
    }
    
    st.markdown(f"## {prompt_titles.get(prompt_type, 'GameMaster AI')}")
    st.caption(f"Game: **{game}** | Type: **{prompt_type.value.replace('_', ' ').title()}**")
    
    # Input area
    if prompt_type != PromptType.COACH_MODE:
        question = st.text_area(
            "Your Question / Context",
            value=context.get("additional_context", "") or context.get("struggles", ""),
            height=150,
            placeholder="Describe what you need help with...",
            key="question_input"
        )
    else:
        question = context.get("struggles", "")
        st.text_area(
            "Your Struggles (from sidebar)",
            value=question,
            height=100,
            disabled=True
        )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🚀 Get Gaming Advice",
            use_container_width=True,
            type="primary"
        )
    
    # Generate response
    if generate_clicked:
        if not gaming_llm.is_model_loaded():
            st.error("❌ Model not loaded! Please load the model from the sidebar first.")
            return
        
        if not question and prompt_type != PromptType.COACH_MODE:
            st.warning("⚠️ Please enter a question or context.")
            return
        
        if prompt_type in [PromptType.BUILD_OPTIMIZER, PromptType.ITEMIZATION, 
                          PromptType.MECHANICS_GUIDE, PromptType.COACH_MODE] and not context.get("champion_hero"):
            st.warning("⚠️ Please specify a champion/hero/character in the sidebar.")
            return
        
        if prompt_type == PromptType.COUNTER_PICK and not context.get("champion_hero"):
            st.warning("⚠️ Please specify an enemy champion/hero in the sidebar.")
            return
        
        # Prepare generation kwargs
        gen_kwargs = {
            "temperature": context.get("temperature", 0.7),
            "max_new_tokens": context.get("max_new_tokens", 2048),
            "top_p": context.get("top_p", 0.9),
        }
        
        # Call appropriate method
        with st.spinner("🎮 GameMaster AI is thinking..."):
            try:
                if prompt_type == PromptType.BUILD_OPTIMIZER:
                    response = gaming_llm.get_build(
                        game=game,
                        champion_hero=context["champion_hero"],
                        role=context.get("role"),
                        player_rank=context.get("player_rank"),
                        patch_version=context.get("patch_version"),
                        additional_context=question,
                        **gen_kwargs
                    )
                elif prompt_type == PromptType.STRATEGY_ADVISOR:
                    response = gaming_llm.get_strategy(
                        game=game,
                        situation=question,
                        player_rank=context.get("player_rank"),
                        role=context.get("role"),
                        patch_version=context.get("patch_version"),
                        **gen_kwargs
                    )
                elif prompt_type == PromptType.LORE_EXPLORER:
                    response = gaming_llm.get_lore(
                        game=game,
                        topic=question,
                        character=context.get("champion_hero"),
                        **gen_kwargs
                    )
                elif prompt_type == PromptType.PATCH_ANALYZER:
                    response = gaming_llm.analyze_patch(
                        game=game,
                        patch_version=context.get("patch_version", "current"),
                        focus=context.get("additional_context"),
                        **gen_kwargs
                    )
                elif prompt_type == PromptType.META_ANALYZER:
                    response = gaming_llm.analyze_meta(
                        game=game,
                        patch_version=context.get("patch_version"),
                        player_rank=context.get("player_rank"),
                        role=context.get("role"),
                        **gen_kwargs
                    )
                elif prompt_type == PromptType.COACH_MODE:
                    response = gaming_llm.get_coaching(
                        game=game,
                        struggles=question,
                        player_rank=context.get("player_rank", "Unranked"),
                        role=context.get("role", "Flex"),
                        champion_hero=context.get("champion_hero"),
                        **gen_kwargs
                    )
                elif prompt_type == PromptType.COUNTER_PICK:
                    response = gaming_llm.get_counters(
                        game=game,
                        enemy_champion=context["champion_hero"],
                        player_rank=context.get("player_rank"),
                        role=context.get("role"),
                        patch_version=context.get("patch_version"),
                        **gen_kwargs
                    )
                elif prompt_type == PromptType.TEAM_COMP:
                    response = gaming_llm.analyze_team_comp(
                        game=game,
                        current_picks=question,
                        player_rank=context.get("player_rank"),
                        patch_version=context.get("patch_version"),
                        **gen_kwargs
                    )
                elif prompt_type == PromptType.ITEMIZATION:
                    response = gaming_llm.get_itemization(
                        game=game,
                        champion_hero=context["champion_hero"],
                        role=context.get("role"),
                        player_rank=context.get("player_rank"),
                        patch_version=context.get("patch_version"),
                        game_state=question,
                        **gen_kwargs
                    )
                elif prompt_type == PromptType.MECHANICS_GUIDE:
                    response = gaming_llm.get_mechanics_guide(
                        game=game,
                        champion_hero=context["champion_hero"],
                        role=context.get("role"),
                        player_rank=context.get("player_rank"),
                        specific_mechanics=question,
                        **gen_kwargs
                    )
                else:  # GENERAL_GAMING
                    response = gaming_llm.ask(
                        game=game,
                        question=question,
                        prompt_type=prompt_type,
                        player_rank=context.get("player_rank"),
                        role=context.get("role"),
                        champion_hero=context.get("champion_hero"),
                        patch_version=context.get("patch_version"),
                        **gen_kwargs
                    )
                
                # Display response
                st.markdown("---")
                st.markdown("### 🎮 Response")
                
                with st.container():
                    st.markdown(f"""
                    <div class="response-box">
                        {response.content}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metadata
                with st.expander("📊 Response Metadata"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prompt Type", response.prompt_type.value.replace('_', ' ').title())
                    with col2:
                        st.metric("Game", response.game)
                    with col3:
                        st.metric("Model", gaming_llm.get_model_info().get('model_id', 'Unknown').split('/')[-1])
                    
                    st.json(response.context.__dict__)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📋 Copy Response", use_container_width=True):
                        st.code(response.content)
                        st.success("Response displayed above for copying!")
                with col2:
                    if st.button("🔄 Regenerate", use_container_width=True):
                        st.rerun()
                with col3:
                    if st.button("💾 Save to History", use_container_width=True):
                        # Could implement save to file/database
                        st.info("Feature coming soon!")
                        
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
                logger.exception("Generation error")


def render_examples():
    """Render example queries in an expander."""
    with st.expander("💡 Example Queries", expanded=False):
        examples = {
            "🏗️ Build Optimizer": [
                "Best build for Jinx ADC in League of Legends patch 14.12",
                "Optimal Pudge build for Dota 2 position 4 support",
                "Jett loadout for Valorant competitive play",
            ],
            "🧠 Strategy Advisor": [
                "How to play from behind as a scaling ADC in League",
                "Early game aggression vs farming on mid lane",
                "How to rotate effectively in Apex Legends ranked",
            ],
            "📜 Lore Explorer": [
                "Explain the Void lore in League of Legends",
                "What is the relationship between Jinx and Vi?",
                "History of the Horde in World of Warcraft",
            ],
            "📋 Patch Analyzer": [
                "Analyze League of Legends patch 14.12 jungle changes",
                "Dota 2 7.35 gameplay changes impact",
                "Valorant patch 8.11 agent balance changes",
            ],
            "📊 Meta Analyzer": [
                "Current League of Legends jungle meta tier list",
                "Dota 2 TI12 meta heroes and strategies",
                "Valorant VCT 2024 meta composition trends",
            ],
            "🎓 Coach Mode": [
                "I'm Gold 2 in League, main ADC, struggle with positioning in teamfights",
                "Hardstuck Platinum in Valorant, need aim training routine",
                "New to Dota 2, coming from League, what should I learn first?",
            ],
            "⚔️ Counter Picks": [
                "Best counters to Yasuo mid lane in League",
                "How to counter Anti-Mage in Dota 2",
                "Counters to Genji in Overwatch 2",
            ],
            "👥 Team Composition": [
                "We have Malphite top, Elise jungle - what mid and bot?",
                "Best deathball composition in Dota 2 current patch",
                "Valorant team comp for Bind map attack side",
            ],
        }
        
        for category, queries in examples.items():
            st.markdown(f"#### {category}")
            for q in queries:
                st.markdown(f"- `{q}`")


def main():
    """Main Streamlit app."""
    render_header()
    
    # Load gaming LLM
    gaming_llm = load_gaming_llm()
    
    if gaming_llm is None:
        st.error("Failed to initialize GameMaster AI. Please check the logs.")
        return
    
    # Render sidebar and get context
    context = render_sidebar(gaming_llm)
    
    # Render main content
    render_main_content(gaming_llm, context)
    
    # Render examples
    render_examples()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎮 GameMaster AI - Your Gaming Companion | Built with ❤️ using Llama 3 & Streamlit</p>
        <p><small>Note: AI responses are for guidance. Always verify with current game resources.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()