"""
CLI Interface for GameMaster AI.
Command-line interface for testing and using the gaming LLM wrapper.
"""

import argparse
import sys
import logging
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint

from .gaming_llm import GamingLLM, create_gaming_llm
from .prompts import PromptType, get_supported_games, get_available_prompt_types

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def print_banner():
    """Print the GameMaster AI banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🎮 GameMaster AI 🎮                        ║
    ║              Your AI Gaming Companion                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_supported_games():
    """Print supported games in a table."""
    games = get_supported_games()
    table = Table(title="Supported Games", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Game", style="cyan")
    table.add_column("Genre", style="green")
    
    genres = {
        "League of Legends": "MOBA",
        "Dota 2": "MOBA",
        "Counter-Strike 2": "FPS",
        "Valorant": "FPS",
        "World of Warcraft": "MMORPG",
        "Final Fantasy XIV": "MMORPG",
        "Elden Ring": "Action RPG",
        "Baldur's Gate 3": "RPG",
        "Cyberpunk 2077": "RPG",
        "The Witcher 3": "RPG",
        "Minecraft": "Sandbox",
        "Roblox": "Sandbox",
        "Fortnite": "Battle Royale",
        "Apex Legends": "Battle Royale",
        "Overwatch 2": "FPS",
    }
    
    for i, game in enumerate(games, 1):
        table.add_row(str(i), game, genres.get(game, "Other"))
    
    console.print(table)


def print_prompt_types():
    """Print available prompt types."""
    types = get_available_prompt_types()
    table = Table(title="Available Prompt Types", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Type", style="cyan")
    table.add_column("Description", style="green")
    
    descriptions = {
        PromptType.BUILD_OPTIMIZER: "Create optimized builds for champions/heroes",
        PromptType.STRATEGY_ADVISOR: "Get strategic advice for game situations",
        PromptType.LORE_EXPLORER: "Explore game lore and story",
        PromptType.PATCH_ANALYZER: "Analyze game patches and changes",
        PromptType.META_ANALYZER: "Analyze current meta and tier lists",
        PromptType.COACH_MODE: "Personalized coaching and improvement plans",
        PromptType.COUNTER_PICK: "Get counter-pick recommendations",
        PromptType.TEAM_COMP: "Analyze and suggest team compositions",
        PromptType.ITEMIZATION: "Detailed itemization guides",
        PromptType.MECHANICS_GUIDE: "Mechanics guides and combos",
        PromptType.GENERAL_GAMING: "General gaming questions",
    }
    
    for i, pt in enumerate(types, 1):
        table.add_row(str(i), pt.value.replace("_", " ").title(), descriptions.get(pt, ""))
    
    console.print(table)


def interactive_mode(gaming_llm: GamingLLM):
    """Run interactive CLI mode."""
    console.print("\n[bold green]Welcome to GameMaster AI Interactive Mode![/bold green]")
    console.print("Type 'help' for commands, 'quit' to exit\n")
    
    # Load model
    with console.status("[bold yellow]Loading model...[/bold yellow]"):
        if not gaming_llm.load_model():
            console.print("[bold red]Failed to load model![/bold red]")
            return
    
    console.print("[bold green]Model loaded successfully![/bold green]\n")
    
    # Show model info
    info = gaming_llm.get_model_info()
    console.print(Panel(
        f"Model: {info.get('model_id', 'Unknown')}\n"
        f"Device: {info.get('device', 'Unknown')}\n"
        f"Quantization: {info.get('quantization', 'Unknown')}",
        title="Model Info",
        border_style="blue"
    ))
    
    while True:
        try:
            # Get game
            game = Prompt.ask("\n[bold cyan]Game[/bold cyan]", default="League of Legends")
            if game.lower() in ['quit', 'exit', 'q']:
                break
            
            # Get prompt type
            console.print("\n[bold]Select prompt type:[/bold]")
            types = get_available_prompt_types()
            for i, pt in enumerate(types, 1):
                console.print(f"  {i}. {pt.value.replace('_', ' ').title()}")
            
            choice = Prompt.ask("Choice", default="1")
            try:
                prompt_type = types[int(choice) - 1]
            except (ValueError, IndexError):
                console.print("[red]Invalid choice, using GENERAL_GAMING[/red]")
                prompt_type = PromptType.GENERAL_GAMING
            
            # Get additional context based on prompt type
            kwargs = {"game": game, "prompt_type": prompt_type}
            
            if prompt_type in [PromptType.BUILD_OPTIMIZER, PromptType.ITEMIZATION, 
                              PromptType.MECHANICS_GUIDE, PromptType.COACH_MODE]:
                kwargs["champion_hero"] = Prompt.ask("Champion/Hero/Character")
                kwargs["role"] = Prompt.ask("Role (optional)", default="")
                if not kwargs["role"]:
                    kwargs["role"] = None
            
            if prompt_type in [PromptType.COUNTER_PICK]:
                kwargs["champion_hero"] = Prompt.ask("Enemy Champion/Hero")
                kwargs["role"] = Prompt.ask("Your Role (optional)", default="")
                if not kwargs["role"]:
                    kwargs["role"] = None
            
            if prompt_type in [PromptType.COACH_MODE]:
                kwargs["player_rank"] = Prompt.ask("Your Rank")
                kwargs["role"] = Prompt.ask("Main Role")
                kwargs["champion_hero"] = Prompt.ask("Main Champion (optional)", default="")
                if not kwargs["champion_hero"]:
                    kwargs["champion_hero"] = None
                kwargs["struggles"] = Prompt.ask("What are you struggling with?")
                del kwargs["prompt_type"]  # Will use get_coaching instead
            
            kwargs["player_rank"] = Prompt.ask("Player Rank (optional)", default="")
            if not kwargs["player_rank"]:
                kwargs["player_rank"] = None
            
            kwargs["patch_version"] = Prompt.ask("Patch Version (optional)", default="")
            if not kwargs["patch_version"]:
                kwargs["patch_version"] = None
            
            # Get question
            if prompt_type != PromptType.COACH_MODE:
                question = Prompt.ask("\n[bold cyan]Your Question[/bold cyan]")
                kwargs["question"] = question
            
            # Generate response
            console.print("\n[bold yellow]Generating response...[/bold yellow]")
            
            try:
                if prompt_type == PromptType.COACH_MODE:
                    response = gaming_llm.get_coaching(
                        game=game,
                        struggles=kwargs["struggles"],
                        player_rank=kwargs["player_rank"],
                        role=kwargs["role"],
                        champion_hero=kwargs.get("champion_hero")
                    )
                else:
                    response = gaming_llm.ask(**kwargs)
                
                # Display response
                console.print("\n")
                console.print(Panel(
                    Markdown(response.content),
                    title=f"🎮 {response.prompt_type.value.replace('_', ' ').title()} - {response.game}",
                    border_style="green",
                    expand=False
                ))
                
            except Exception as e:
                console.print(f"[bold red]Error: {e}[/bold red]")
                logger.exception("Generation error")
            
            # Continue?
            if not Confirm.ask("\n[bold]Ask another question?[/bold]", default=True):
                break
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            break
        except EOFError:
            break
    
    console.print("\n[bold cyan]Thanks for using GameMaster AI! Good luck in your games! 🎮[/bold cyan]")


def single_question_mode(gaming_llm: GamingLLM, args):
    """Run single question mode."""
    with console.status("[bold yellow]Loading model...[/bold yellow]"):
        if not gaming_llm.load_model():
            console.print("[bold red]Failed to load model![/bold red]")
            sys.exit(1)
    
    console.print("[bold green]Model loaded![/bold green]\n")
    
    # Build kwargs
    kwargs = {
        "game": args.game,
        "question": args.question,
        "prompt_type": PromptType(args.type),
        "player_rank": args.rank,
        "role": args.role,
        "champion_hero": args.champion,
        "patch_version": args.patch,
    }
    
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    try:
        response = gaming_llm.ask(**kwargs)
        
        console.print(Panel(
            Markdown(response.content),
            title=f"🎮 {response.prompt_type.value.replace('_', ' ').title()} - {response.game}",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        logger.exception("Generation error")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GameMaster AI - Your Gaming Companion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m src.cli --interactive
  
  # Single question
  python -m src.cli --game "League of Legends" --question "Best build for Jinx" --type build_optimizer --champion "Jinx" --role "ADC"
  
  # Get coaching
  python -m src.cli --game "Valorant" --type coach_mode --rank "Gold 2" --role "Duelist" --champion "Jett" --question "How to improve aim"
  
  # List supported games
  python -m src.cli --list-games
  
  # List prompt types
  python -m src.cli --list-types
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to config YAML file",
        default="config/config.yaml"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--game", "-g",
        help="Game name",
        default="League of Legends"
    )
    parser.add_argument(
        "--question", "-q",
        help="Question or context"
    )
    parser.add_argument(
        "--type", "-t",
        help="Prompt type",
        choices=[pt.value for pt in get_available_prompt_types()],
        default="general_gaming"
    )
    parser.add_argument(
        "--rank", "-r",
        help="Player rank/skill level"
    )
    parser.add_argument(
        "--role",
        help="Player role"
    )
    parser.add_argument(
        "--champion", "--champ",
        help="Champion/hero/character name"
    )
    parser.add_argument(
        "--patch", "-p",
        help="Game patch version"
    )
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="List supported games"
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List available prompt types"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Print banner
    print_banner()
    
    # Handle list commands
    if args.list_games:
        print_supported_games()
        return
    
    if args.list_types:
        print_prompt_types()
        return
    
    # Create gaming LLM
    gaming_llm = create_gaming_llm(config_path=args.config)
    
    # Run appropriate mode
    if args.interactive or (not args.question and not args.list_games and not args.list_types):
        interactive_mode(gaming_llm)
    else:
        if not args.question:
            parser.error("--question is required for single question mode")
        single_question_mode(gaming_llm, args)


if __name__ == "__main__":
    main()