"""
Gradio Web UI for GameMaster AI.
Alternative web interface using Gradio.
"""

import gradio as gr
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import sys

# Import our gaming LLM
sys.path.append(str(Path(__file__).parent.parent))
from src.gaming_llm import GamingLLM, create_gaming_llm
from src.prompts import PromptType, get_supported_games, get_available_prompt_types, detect_game_genre

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global gaming LLM instance
gaming_llm: Optional[GamingLLM] = None


def initialize_gaming_llm(config_path: str = "config/config.yaml") -> GamingLLM:
    """Initialize the Gaming LLM."""
    global gaming_llm
    if gaming_llm is None:
        gaming_llm = create_gaming_llm(config_path=config_path)
    return gaming_llm


def load_model_fn() -> str:
    """Load the model."""
    global gaming_llm
    if gaming_llm is None:
        gaming_llm = initialize_gaming_llm()
    
    if gaming_llm.is_model_loaded():
        return "✅ Model already loaded!"
    
    success = gaming_llm.load_model()
    if success:
        info = gaming_llm.get_model_info()
        return f"✅ Model loaded successfully!\nModel: {info.get('model_id', 'Unknown')}\nDevice: {info.get('device', 'Unknown')}\nQuantization: {info.get('quantization', 'Unknown')}"
    else:
        return "❌ Failed to load model. Check logs for details."


def get_model_status() -> str:
    """Get model status."""
    global gaming_llm
    if gaming_llm is None:
        gaming_llm = initialize_gaming_llm()
    
    if gaming_llm.is_model_loaded():
        info = gaming_llm.get_model_info()
        return f"✅ **Model Loaded**\n- Model: {info.get('model_id', 'Unknown')}\n- Device: {info.get('device', 'Unknown')}\n- Quantization: {info.get('quantization', 'Unknown')}"
    else:
        return "⚠️ **Model Not Loaded**\nClick 'Load Model' to initialize."


def generate_response(
    game: str,
    prompt_type: str,
    question: str,
    player_rank: str,
    role: str,
    champion_hero: str,
    patch_version: str,
    struggles: str,
    additional_context: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float
) -> str:
    """Generate a gaming response."""
    global gaming_llm
    if gaming_llm is None:
        gaming_llm = initialize_gaming_llm()
    
    if not gaming_llm.is_model_loaded():
        return "❌ Model not loaded! Please click 'Load Model' first."
    
    try:
        pt = PromptType(prompt_type)
        gen_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
        }
        
        # Clean up empty strings
        player_rank = player_rank or None
        role = role or None
        champion_hero = champion_hero or None
        patch_version = patch_version or None
        struggles = struggles or None
        additional_context = additional_context or None
        question = question or None
        
        if pt == PromptType.BUILD_OPTIMIZER:
            response = gaming_llm.get_build(
                game=game,
                champion_hero=champion_hero,
                role=role,
                player_rank=player_rank,
                patch_version=patch_version,
                additional_context=question or additional_context,
                **gen_kwargs
            )
        elif pt == PromptType.STRATEGY_ADVISOR:
            response = gaming_llm.get_strategy(
                game=game,
                situation=question or additional_context,
                player_rank=player_rank,
                role=role,
                patch_version=patch_version,
                **gen_kwargs
            )
        elif pt == PromptType.LORE_EXPLORER:
            response = gaming_llm.get_lore(
                game=game,
                topic=question or additional_context,
                character=champion_hero,
                **gen_kwargs
            )
        elif pt == PromptType.PATCH_ANALYZER:
            response = gaming_llm.analyze_patch(
                game=game,
                patch_version=patch_version or "current",
                focus=additional_context,
                **gen_kwargs
            )
        elif pt == PromptType.META_ANALYZER:
            response = gaming_llm.analyze_meta(
                game=game,
                patch_version=patch_version,
                player_rank=player_rank,
                role=role,
                **gen_kwargs
            )
        elif pt == PromptType.COACH_MODE:
            response = gaming_llm.get_coaching(
                game=game,
                struggles=struggles or question or additional_context,
                player_rank=player_rank or "Unranked",
                role=role or "Flex",
                champion_hero=champion_hero,
                **gen_kwargs
            )
        elif pt == PromptType.COUNTER_PICK:
            response = gaming_llm.get_counters(
                game=game,
                enemy_champion=champion_hero,
                player_rank=player_rank,
                role=role,
                patch_version=patch_version,
                **gen_kwargs
            )
        elif pt == PromptType.TEAM_COMP:
            response = gaming_llm.analyze_team_comp(
                game=game,
                current_picks=question or additional_context,
                player_rank=player_rank,
                patch_version=patch_version,
                **gen_kwargs
            )
        elif pt == PromptType.ITEMIZATION:
            response = gaming_llm.get_itemization(
                game=game,
                champion_hero=champion_hero,
                role=role,
                player_rank=player_rank,
                patch_version=patch_version,
                game_state=question or additional_context,
                **gen_kwargs
            )
        elif pt == PromptType.MECHANICS_GUIDE:
            response = gaming_llm.get_mechanics_guide(
                game=game,
                champion_hero=champion_hero,
                role=role,
                player_rank=player_rank,
                specific_mechanics=question or additional_context,
                **gen_kwargs
            )
        else:  # GENERAL_GAMING
            response = gaming_llm.ask(
                game=game,
                question=question or additional_context,
                prompt_type=pt,
                player_rank=player_rank,
                role=role,
                champion_hero=champion_hero,
                patch_version=patch_version,
                **gen_kwargs
            )
        
        return response.content
        
    except Exception as e:
        logger.exception("Generation error")
        return f"❌ Error: {str(e)}"


def clear_history_fn() -> str:
    """Clear conversation history."""
    global gaming_llm
    if gaming_llm:
        gaming_llm.clear_history()
    return "✅ Conversation history cleared!"


def create_gradio_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    
    # Get supported games and prompt types
    games = get_supported_games()
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
    prompt_type_choices = [(prompt_type_labels[pt], pt.value) for pt in prompt_types]
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .status-loaded {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-not-loaded {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    """
    
    with gr.Blocks(css=css, title="🎮 GameMaster AI", theme=gr.themes.Soft()) as demo:
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎮 GameMaster AI</h1>
            <p>Your Intelligent Gaming Companion - Builds, Strategies, Lore & Coaching</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                # Model Status
                with gr.Group():
                    gr.Markdown("### 🤖 Model Status")
                    model_status = gr.Markdown(
                        value="⚠️ **Model Not Loaded**\nClick 'Load Model' to initialize.",
                        elem_classes=["status-box", "status-not-loaded"]
                    )
                    load_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")
                    load_btn.click(
                        fn=load_model_fn,
                        outputs=model_status
                    )
                
                gr.Markdown("---")
                
                # Game Selection
                with gr.Group():
                    gr.Markdown("### 🎮 Game Selection")
                    game_dropdown = gr.Dropdown(
                        choices=games,
                        value=games[0],
                        label="Select Game",
                        interactive=True
                    )
                    genre_display = gr.Markdown("Genre: MOBA")
                    
                    def update_genre(game):
                        genre = detect_game_genre(game)
                        return f"Genre: {genre.value.upper()}"
                    
                    game_dropdown.change(
                        fn=update_genre,
                        inputs=game_dropdown,
                        outputs=genre_display
                    )
                
                # Prompt Type
                with gr.Group():
                    gr.Markdown("### 🎯 Assistance Type")
                    prompt_type_dropdown = gr.Dropdown(
                        choices=prompt_type_choices,
                        value=prompt_type_choices[0][1],
                        label="What do you need?",
                        interactive=True
                    )
                
                # Context Inputs (dynamic based on prompt type)
                with gr.Group():
                    gr.Markdown("### 📝 Context")
                    
                    # Common fields
                    player_rank = gr.Textbox(
                        label="🏆 Your Rank (optional)",
                        placeholder="e.g., Gold 2, Diamond 1, Level 50"
                    )
                    patch_version = gr.Textbox(
                        label="📦 Patch Version (optional)",
                        placeholder="e.g., 14.12, 7.35, Season 12"
                    )
                    
                    # Dynamic fields container
                    with gr.Group(visible=True) as role_group:
                        role = gr.Textbox(
                            label="🎭 Role (optional)",
                            placeholder="e.g., ADC, Mid, Top, Jungle, Support"
                        )
                    
                    with gr.Group(visible=True) as champion_group:
                        champion_hero = gr.Textbox(
                            label="⚔️ Champion/Hero/Character",
                            placeholder="e.g., Jinx, Pudge, Jett, Reaper"
                        )
                    
                    with gr.Group(visible=False) as enemy_champion_group:
                        enemy_champion = gr.Textbox(
                            label="🎯 Enemy Champion/Hero",
                            placeholder="e.g., Yasuo, Anti-Mage, Genji"
                        )
                    
                    with gr.Group(visible=False) as struggles_group:
                        struggles = gr.Textbox(
                            label="🎯 What are you struggling with?",
                            placeholder="e.g., 'I lose lane against assassins', 'My teamfight positioning is bad'",
                            lines=3
                        )
                    
                    with gr.Group(visible=False) as team_comp_group:
                        team_comp = gr.Textbox(
                            label="👥 Current Team Picks / Situation",
                            placeholder="e.g., 'We have Malphite top, Elise jungle, need mid and bot'",
                            lines=3
                        )
                    
                    with gr.Group(visible=False) as strategy_group:
                        strategy = gr.Textbox(
                            label="🧠 Situation Description",
                            placeholder="e.g., 'How to play from behind as ADC', 'Early game aggression vs scaling'",
                            lines=3
                        )
                    
                    with gr.Group(visible=False) as lore_group:
                        lore_character = gr.Textbox(
                            label="👤 Character/Faction/Location (optional)",
                            placeholder="e.g., Jinx, The Void, Runeterra"
                        )
                        lore_topic = gr.Textbox(
                            label="📜 Lore Topic",
                            placeholder="e.g., 'Origin of the Void', 'Relationship between Jinx and Vi'",
                            lines=3
                        )
                    
                    with gr.Group(visible=False) as patch_focus_group:
                        patch_focus = gr.Textbox(
                            label="🔍 Focus Area (optional)",
                            placeholder="e.g., 'Jungle changes', 'Item reworks'",
                            lines=2
                        )
                    
                    with gr.Group(visible=False) as meta_role_group:
                        meta_role = gr.Textbox(
                            label="🎭 Role Focus (optional)",
                            placeholder="e.g., Jungle, Support, All roles"
                        )
                    
                    with gr.Group(visible=True) as general_group:
                        general_question = gr.Textbox(
                            label="❓ Your Question",
                            placeholder="Ask anything about the game...",
                            lines=3
                        )
                
                # Advanced Settings
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    max_new_tokens = gr.Slider(
                        minimum=512, maximum=4096, value=2048, step=256,
                        label="Max Response Length"
                    )
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                        label="Top-p"
                    )
                
                # Clear History
                clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                clear_status = gr.Markdown()
                clear_btn.click(
                    fn=clear_history_fn,
                    outputs=clear_status
                )
            
            # Right column - Chat/Response
            with gr.Column(scale=2):
                # Question Input
                with gr.Group():
                    gr.Markdown("### 💬 Your Question")
                    question_input = gr.Textbox(
                        label="",
                        placeholder="Describe what you need help with...",
                        lines=5
                    )
                
                # Generate Button
                generate_btn = gr.Button("🚀 Get Gaming Advice", variant="primary", size="lg")
                
                # Response Output
                with gr.Group():
                    gr.Markdown("### 🎮 Response")
                    response_output = gr.Markdown(
                        value="Your gaming advice will appear here...",
                        elem_classes=["response-box"]
                    )
                
                # Examples
                with gr.Accordion("💡 Example Queries", open=False):
                    gr.Markdown("""
                    **🏗️ Build Optimizer**
                    - Best build for Jinx ADC in League of Legends patch 14.12
                    - Optimal Pudge build for Dota 2 position 4 support
                    - Jett loadout for Valorant competitive play
                    
                    **🧠 Strategy Advisor**
                    - How to play from behind as a scaling ADC in League
                    - Early game aggression vs farming on mid lane
                    - How to rotate effectively in Apex Legends ranked
                    
                    **📜 Lore Explorer**
                    - Explain the Void lore in League of Legends
                    - What is the relationship between Jinx and Vi?
                    - History of the Horde in World of Warcraft
                    
                    **📋 Patch Analyzer**
                    - Analyze League of Legends patch 14.12 jungle changes
                    - Dota 2 7.35 gameplay changes impact
                    - Valorant patch 8.11 agent balance changes
                    
                    **📊 Meta Analyzer**
                    - Current League of Legends jungle meta tier list
                    - Dota 2 TI12 meta heroes and strategies
                    - Valorant VCT 2024 meta composition trends
                    
                    **🎓 Coach Mode**
                    - I'm Gold 2 in League, main ADC, struggle with positioning in teamfights
                    - Hardstuck Platinum in Valorant, need aim training routine
                    - New to Dota 2, coming from League, what should I learn first?
                    
                    **⚔️ Counter Picks**
                    - Best counters to Yasuo mid lane in League
                    - How to counter Anti-Mage in Dota 2
                    - Counters to Genji in Overwatch 2
                    
                    **👥 Team Composition**
                    - We have Malphite top, Elise jungle - what mid and bot?
                    - Best deathball composition in Dota 2 current patch
                    - Valorant team comp for Bind map attack side
                    """)
        
        # Dynamic visibility based on prompt type
        def update_visibility(prompt_type):
            pt = PromptType(prompt_type)
            return {
                role_group: gr.Group(visible=pt in [PromptType.BUILD_OPTIMIZER, PromptType.ITEMIZATION, 
                                                    PromptType.MECHANICS_GUIDE, PromptType.COACH_MODE,
                                                    PromptType.COUNTER_PICK, PromptType.STRATEGY_ADVISOR,
                                                    PromptType.TEAM_COMP, PromptType.META_ANALYZER]),
                champion_group: gr.Group(visible=pt in [PromptType.BUILD_OPTIMIZER, PromptType.ITEMIZATION,
                                                        PromptType.MECHANICS_GUIDE, PromptType.COACH_MODE]),
                enemy_champion_group: gr.Group(visible=pt == PromptType.COUNTER_PICK),
                struggles_group: gr.Group(visible=pt == PromptType.COACH_MODE),
                team_comp_group: gr.Group(visible=pt == PromptType.TEAM_COMP),
                strategy_group: gr.Group(visible=pt == PromptType.STRATEGY_ADVISOR),
                lore_group: gr.Group(visible=pt == PromptType.LORE_EXPLORER),
                patch_focus_group: gr.Group(visible=pt == PromptType.PATCH_ANALYZER),
                meta_role_group: gr.Group(visible=pt == PromptType.META_ANALYZER),
                general_group: gr.Group(visible=pt == PromptType.GENERAL_GAMING),
            }
        
        prompt_type_dropdown.change(
            fn=update_visibility,
            inputs=prompt_type_dropdown,
            outputs=[role_group, champion_group, enemy_champion_group, struggles_group,
                    team_comp_group, strategy_group, lore_group, patch_focus_group,
                    meta_role_group, general_group]
        )
        
        # Generate response
        def generate_wrapper(
            game, prompt_type, question, player_rank, role, champion_hero, patch_version,
            struggles, additional_context, temperature, max_new_tokens, top_p
        ):
            # Determine which context field to use based on prompt type
            pt = PromptType(prompt_type)
            ctx = ""
            if pt == PromptType.COACH_MODE:
                ctx = struggles
            elif pt == PromptType.TEAM_COMP:
                ctx = additional_context
            elif pt == PromptType.STRATEGY_ADVISOR:
                ctx = additional_context
            elif pt == PromptType.LORE_EXPLORER:
                ctx = additional_context
            elif pt == PromptType.PATCH_ANALYZER:
                ctx = additional_context
            elif pt == PromptType.META_ANALYZER:
                ctx = additional_context
            elif pt == PromptType.GENERAL_GAMING:
                ctx = additional_context
            else:
                ctx = question
            
            return generate_response(
                game=game,
                prompt_type=prompt_type,
                question=question,
                player_rank=player_rank,
                role=role,
                champion_hero=champion_hero,
                patch_version=patch_version,
                struggles=struggles,
                additional_context=ctx,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )
        
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[
                game_dropdown, prompt_type_dropdown, question_input,
                player_rank, role, champion_hero, patch_version,
                struggles, general_question,  # Using general_question as additional_context fallback
                temperature, max_new_tokens, top_p
            ],
            outputs=response_output
        )
        
        # Footer
        gr.Markdown("""
        ---
        <div style='text-align: center; color: #666;'>
            <p>🎮 GameMaster AI - Your Gaming Companion | Built with ❤️ using Llama 3 & Gradio</p>
            <p><small>Note: AI responses are for guidance. Always verify with current game resources.</small></p>
        </div>
        """)
    
    return demo


def main():
    """Launch the Gradio app."""
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()