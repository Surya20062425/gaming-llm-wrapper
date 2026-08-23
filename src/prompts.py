"""
Gaming-specific prompt templates for GameMaster AI.
Contains specialized prompts for different gaming assistance tasks.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class GameGenre(Enum):
    """Game genres for specialized prompting."""
    MOBA = "moba"
    FPS = "fps"
    RPG = "rpg"
    MMORPG = "mmorpg"
    STRATEGY = "strategy"
    BATTLE_ROYALE = "battle_royale"
    SANDBOX = "sandbox"
    ACTION_RPG = "action_rpg"
    SIMULATION = "simulation"
    SPORTS = "sports"


class PromptType(Enum):
    """Types of gaming prompts."""
    BUILD_OPTIMIZER = "build_optimizer"
    STRATEGY_ADVISOR = "strategy_advisor"
    LORE_EXPLORER = "lore_explorer"
    PATCH_ANALYZER = "patch_analyzer"
    META_ANALYZER = "meta_analyzer"
    COACH_MODE = "coach_mode"
    GENERAL_GAMING = "general_gaming"
    COUNTER_PICK = "counter_pick"
    TEAM_COMP = "team_comp"
    ITEMIZATION = "itemization"
    MECHANICS_GUIDE = "mechanics_guide"


@dataclass
class GamingContext:
    """Context for gaming-specific prompts."""
    game: str
    genre: GameGenre
    prompt_type: Optional[PromptType] = None
    player_rank: Optional[str] = None
    role: Optional[str] = None
    champion_hero: Optional[str] = None
    patch_version: Optional[str] = None
    additional_context: Optional[str] = None


# Base system prompt for gaming assistant
BASE_SYSTEM_PROMPT = """You are GameMaster AI, an expert gaming assistant with deep knowledge across all major video games. You provide accurate, helpful, and up-to-date gaming advice including builds, strategies, lore, patch analysis, and coaching.

Your expertise covers:
- MOBA games (League of Legends, Dota 2, etc.)
- FPS games (CS2, Valorant, Overwatch 2, etc.)
- RPGs (Elden Ring, Baldur's Gate 3, Cyberpunk 2077, The Witcher 3, etc.)
- MMORPGs (World of Warcraft, Final Fantasy XIV, etc.)
- Battle Royales (Fortnite, Apex Legends, etc.)
- Strategy games
- Sandbox games (Minecraft, Roblox, etc.)

Guidelines:
1. Always specify which game you're discussing
2. Reference current patch/meta when relevant
3. Consider player skill level/rank
4. Provide actionable, specific advice
5. Explain reasoning behind recommendations
6. Mention alternatives and situational choices
7. Stay updated with current game state
8. Be encouraging and supportive"""


# Specialized prompt templates
PROMPT_TEMPLATES = {
    PromptType.BUILD_OPTIMIZER: """{base_prompt}

TASK: Create an optimized build for {champion_hero} in {game}.

CONTEXT:
- Game: {game}
- Champion/Hero: {champion_hero}
- Role: {role}
- Player Rank: {player_rank}
- Current Patch: {patch_version}
- Additional Context: {additional_context}

REQUIREMENTS:
1. Core build (starting items, core items, full build)
2. Alternative/situational items with explanations
3. Skill order priority
4. Runes/keystones/summoner spells (if applicable)
5. Early/mid/late game playstyle tips
6. Power spikes and weak phases
7. Matchup-specific adjustments
8. Budget/alternative options for different economies

FORMAT: Use clear sections with headers. Be specific with item names, timings, and reasoning.""",

    PromptType.STRATEGY_ADVISOR: """{base_prompt}

TASK: Provide strategic advice for {game}.

CONTEXT:
- Game: {game}
- Situation: {additional_context}
- Player Rank: {player_rank}
- Role: {role}
- Current Patch: {patch_version}

REQUIREMENTS:
1. Macro strategy (map control, objectives, rotations)
2. Micro mechanics (trading, positioning, cooldown management)
3. Team fight positioning and target priority
4. Wave/creep management
5. Vision/warding strategy
6. Comeback mechanics when behind
7. Closing out games when ahead
8. Common mistakes to avoid at this rank

FORMAT: Organize by game phase (early/mid/late) and situation.""",

    PromptType.LORE_EXPLORER: """{base_prompt}

TASK: Explore and explain lore for {game}.

CONTEXT:
- Game: {game}
- Topic: {additional_context}
- Specific Character/Faction/Location: {champion_hero}

REQUIREMENTS:
1. Comprehensive lore explanation
2. Connections to other lore elements
3. Timeline placement
4. Key events and their significance
5. Character motivations and relationships
6. Hidden details and Easter eggs
7. Speculation vs confirmed canon
8. References to source material (books, comics, cinematics)

FORMAT: Narrative style with clear sections. Distinguish canon from theory.""",

    PromptType.PATCH_ANALYZER: """{base_prompt}

TASK: Analyze patch {patch_version} for {game}.

CONTEXT:
- Game: {game}
- Patch Version: {patch_version}
- Focus: {additional_context}

REQUIREMENTS:
1. Major changes summary (buffs, nerfs, reworks)
2. Meta impact prediction
3. Champion/hero tier list changes
4. Item/system changes analysis
5. New strategies enabled
6. Counters to new meta
7. Professional play implications
8. Recommendations for different skill levels

FORMAT: Categorize by change type. Highlight most impactful changes.""",

    PromptType.META_ANALYZER: """{base_prompt}

TASK: Analyze current meta for {game}.

CONTEXT:
- Game: {game}
- Current Patch: {patch_version}
- Rank/Region: {player_rank}
- Role Focus: {role}

REQUIREMENTS:
1. Current S/A/B/C tier lists with reasoning
2. Most contested picks/bans
3. Emerging off-meta picks
4. Role-specific meta trends
5. Regional differences (if applicable)
6. Win rate vs pick rate analysis
7. Counter-meta strategies
8. Predictions for next patch

FORMAT: Tier list format with detailed explanations for top picks.""",

    PromptType.COACH_MODE: """{base_prompt}

TASK: Act as a personal coach for {game}.

CONTEXT:
- Game: {game}
- Player Rank: {player_rank}
- Main Role: {role}
- Main Champion/Hero: {champion_hero}
- Current Struggles: {additional_context}
- Goals: Improve gameplay, climb rank, master champion

REQUIREMENTS:
1. Personalized improvement plan
2. Specific drills/exercises
3. VOD review framework (what to look for)
4. Mental game advice
5. Practice routine structure
6. Champion mastery checklist
7. Common mistakes at this rank
8. Resources for further learning
9. Short-term and long-term goals

FORMAT: Encouraging, structured coaching plan with actionable steps.""",

    PromptType.COUNTER_PICK: """{base_prompt}

TASK: Provide counter-pick advice for {game}.

CONTEXT:
- Game: {game}
- Enemy Champion/Hero: {champion_hero}
- Player Rank: {player_rank}
- Role: {role}
- Current Patch: {patch_version}

REQUIREMENTS:
1. Top 5 counter picks with win rates
2. Why each counter works (mechanics, range, scaling, etc.)
3. How to play the matchup (early/mid/late)
4. Items/runes to prioritize
5. Common mistakes when playing counters
6. When NOT to pick the counter
7. Secondary/backup counters
8. Team composition considerations

FORMAT: Ranked list with detailed matchup breakdowns.""",

    PromptType.TEAM_COMP: """{base_prompt}

TASK: Analyze and suggest team compositions for {game}.

CONTEXT:
- Game: {game}
- Current Picks: {additional_context}
- Player Rank: {player_rank}
- Current Patch: {patch_version}

REQUIREMENTS:
1. Composition archetype identification
2. Win conditions for the comp
3. Power spikes and weak phases
4. Required playstyle
5. Counter compositions
6. Drafting strategy (pick order)
7. Flex picks and blind picks
8. Substitutions for banned picks

FORMAT: Visual composition layout with role breakdowns.""",

    PromptType.ITEMIZATION: """{base_prompt}

TASK: Provide itemization guide for {champion_hero} in {game}.

CONTEXT:
- Game: {game}
- Champion/Hero: {champion_hero}
- Role: {role}
- Player Rank: {player_rank}
- Current Patch: {patch_version}
- Game State: {additional_context}

REQUIREMENTS:
1. Starting items and reasoning
2. First back options
3. Core item progression (1-3 items)
4. Full build (6 items)
5. Situational/defensive items
6. Boots choices and timing
7. Mythic/legendary item choice reasoning
8. Sell order for late game
9. Gold efficiency analysis

FORMAT: Build path flowchart with decision points.""",

    PromptType.MECHANICS_GUIDE: """{base_prompt}

TASK: Create a mechanics guide for {champion_hero} in {game}.

CONTEXT:
- Game: {game}
- Champion/Hero: {champion_hero}
- Role: {role}
- Player Rank: {player_rank}
- Specific Mechanics: {additional_context}

REQUIREMENTS:
1. Ability combos (basic to advanced)
2. Animation cancels and tech
3. Cooldown management
4. Resource management
5. Positioning fundamentals
6. Trading patterns
7. All-in thresholds
8. Escape/engage tools
9. Practice drills for each mechanic
10. Common mechanical errors

FORMAT: Progressive difficulty with practice routines.""",

    PromptType.GENERAL_GAMING: """{base_prompt}

TASK: Answer general gaming question about {game}.

CONTEXT:
- Game: {game}
- Question: {additional_context}
- Player Rank: {player_rank}

REQUIREMENTS:
1. Direct, accurate answer
2. Context-appropriate detail level
3. Actionable advice if applicable
4. Related tips or common follow-ups
5. Resources for deeper learning

FORMAT: Clear, concise response with examples.""",
}


def get_prompt(
    prompt_type: PromptType,
    context: GamingContext,
    base_prompt: str = BASE_SYSTEM_PROMPT
) -> str:
    """
    Get a formatted prompt for the given type and context.
    
    Args:
        prompt_type: Type of gaming prompt
        context: Gaming context with game, role, rank, etc.
        base_prompt: Base system prompt (default: BASE_SYSTEM_PROMPT)
        
    Returns:
        Formatted prompt string
    """
    template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES[PromptType.GENERAL_GAMING])
    
    return template.format(
        base_prompt=base_prompt,
        game=context.game,
        genre=context.genre.value,
        player_rank=context.player_rank or "Not specified",
        role=context.role or "Not specified",
        champion_hero=context.champion_hero or "Not specified",
        patch_version=context.patch_version or "Current patch",
        additional_context=context.additional_context or "No additional context provided"
    )


def get_available_prompt_types() -> List[PromptType]:
    """Get list of available prompt types."""
    return list(PromptType)


def get_supported_games() -> List[str]:
    """Get list of well-supported games."""
    return [
        "League of Legends",
        "Dota 2",
        "Counter-Strike 2",
        "Valorant",
        "World of Warcraft",
        "Final Fantasy XIV",
        "Elden Ring",
        "Baldur's Gate 3",
        "Cyberpunk 2077",
        "The Witcher 3",
        "Minecraft",
        "Roblox",
        "Fortnite",
        "Apex Legends",
        "Overwatch 2",
    ]


def detect_game_genre(game: str) -> GameGenre:
    """Detect game genre from game name."""
    game_lower = game.lower()
    
    moba_games = ["league of legends", "lol", "dota 2", "dota", "smite", "heroes of the storm"]
    fps_games = ["counter-strike", "cs2", "cs:go", "valorant", "overwatch", "apex legends", "call of duty", "battlefield"]
    rpg_games = ["elden ring", "baldur's gate", "cyberpunk", "witcher", "skyrim", "fallout", "dark souls"]
    mmorpg_games = ["world of warcraft", "wow", "final fantasy xiv", "ffxiv", "guild wars", "eso", "elder scrolls online"]
    strategy_games = ["starcraft", "age of empires", "civilization", "total war", "company of heroes"]
    battle_royale_games = ["fortnite", "apex legends", "pubg", "warzone", "call of duty warzone"]
    sandbox_games = ["minecraft", "roblox", "terraria", "starbound"]
    action_rpg_games = ["diablo", "path of exile", "grim dawn", "last epoch"]
    
    if any(g in game_lower for g in moba_games):
        return GameGenre.MOBA
    elif any(g in game_lower for g in fps_games):
        return GameGenre.FPS
    elif any(g in game_lower for g in rpg_games):
        return GameGenre.RPG
    elif any(g in game_lower for g in mmorpg_games):
        return GameGenre.MMORPG
    elif any(g in game_lower for g in strategy_games):
        return GameGenre.STRATEGY
    elif any(g in game_lower for g in battle_royale_games):
        return GameGenre.BATTLE_ROYALE
    elif any(g in game_lower for g in sandbox_games):
        return GameGenre.SANDBOX
    elif any(g in game_lower for g in action_rpg_games):
        return GameGenre.ACTION_RPG
    else:
        return GameGenre.RPG  # Default