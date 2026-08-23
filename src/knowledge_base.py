"""
Knowledge Base for GameMaster AI.
Implements RAG (Retrieval-Augmented Generation) for gaming-specific knowledge.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG dependencies not available. Install sentence-transformers and chromadb for knowledge base.")


@dataclass
class GamingDocument:
    """Represents a gaming knowledge document."""
    id: str
    content: str
    metadata: Dict[str, Any]
    game: str
    category: str  # build, strategy, lore, patch, meta, mechanics, general


class GamingKnowledgeBase:
    """
    Knowledge base for gaming-specific information using RAG.
    Stores and retrieves gaming guides, patch notes, builds, strategies, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the knowledge base.
        
        Args:
            config: Knowledge base configuration
        """
        self.config = config
        self.kb_config = config.get("knowledge_base", {})
        self.enabled = self.kb_config.get("enabled", False) and RAG_AVAILABLE
        
        if not self.enabled:
            logger.info("Knowledge base disabled or dependencies not available")
            self.client = None
            self.collection = None
            self.embedding_model = None
            return
        
        # Initialize embedding model
        embedding_model_name = self.kb_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB
        persist_dir = self.kb_config.get("data_path", "./data/gaming_knowledge")
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="gaming_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Knowledge base initialized at {persist_dir}")
        logger.info(f"Documents in collection: {self.collection.count()}")
    
    def add_document(
        self,
        content: str,
        game: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document content
            game: Game name
            category: Category (build, strategy, lore, patch, meta, mechanics, general)
            metadata: Additional metadata
            doc_id: Optional document ID
            
        Returns:
            Document ID
        """
        if not self.enabled:
            logger.warning("Knowledge base not enabled")
            return ""
        
        import uuid
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Prepare metadata
        doc_metadata = {
            "game": game,
            "category": category,
            **(metadata or {})
        }
        
        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()
        
        # Add to collection
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[doc_metadata],
            embeddings=[embedding]
        )
        
        logger.info(f"Added document {doc_id} for {game} ({category})")
        return doc_id
    
    def add_documents_batch(self, documents: List[GamingDocument]) -> List[str]:
        """Add multiple documents at once."""
        if not self.enabled:
            return []
        
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [
            {"game": doc.game, "category": doc.category, **doc.metadata}
            for doc in documents
        ]
        embeddings = self.embedding_model.encode(contents).tolist()
        
        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
        return ids
    
    def search(
        self,
        query: str,
        game: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            game: Filter by game
            category: Filter by category
            top_k: Number of results
            
        Returns:
            List of matching documents with scores
        """
        if not self.enabled:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Build where filter
        where_filter = {}
        if game:
            where_filter["game"] = game
        if category:
            where_filter["category"] = category
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def get_relevant_context(
        self,
        query: str,
        game: str,
        prompt_type: str,
        top_k: int = 3
    ) -> str:
        """
        Get relevant context for a gaming query.
        
        Args:
            query: User query
            game: Game name
            prompt_type: Type of prompt (build, strategy, etc.)
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        if not self.enabled:
            return ""
        
        # Map prompt type to categories
        category_map = {
            "build_optimizer": ["build", "itemization", "runes"],
            "strategy_advisor": ["strategy", "macro", "micro", "gameplay"],
            "lore_explorer": ["lore", "story", "character", "history"],
            "patch_analyzer": ["patch", "changelog", "balance", "update"],
            "meta_analyzer": ["meta", "tier", "pickrate", "winrate", "pro"],
            "coach_mode": ["guide", "tips", "improvement", "coaching", "mechanics"],
            "counter_pick": ["counter", "matchup", "versus", "build"],
            "team_comp": ["composition", "team", "synergy", "draft"],
            "itemization": ["item", "build", "itemization", "shop"],
            "mechanics_guide": ["mechanics", "combo", "technique", "animation"],
            "general_gaming": ["general", "guide", "tips", "beginner"]
        }
        
        categories = category_map.get(prompt_type, ["general"])
        
        all_results = []
        for cat in categories:
            results = self.search(query, game=game, category=cat, top_k=top_k)
            all_results.extend(results)
        
        # Sort by score and take top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:top_k]
        
        if not top_results:
            return ""
        
        # Format context
        context_parts = ["=== RELEVANT GAMING KNOWLEDGE ==="]
        for i, result in enumerate(top_results, 1):
            context_parts.append(f"\n--- Source {i} ({result['metadata'].get('category', 'unknown')}) ---")
            context_parts.append(result["content"][:1500])  # Limit length
        
        context_parts.append("\n=== END KNOWLEDGE ===")
        return "\n".join(context_parts)
    
    def load_from_directory(self, directory: str) -> int:
        """
        Load documents from a directory structure.
        
        Expected structure:
        directory/
            game_name/
                category/
                    *.txt, *.md files
        
        Args:
            directory: Path to knowledge directory
            
        Returns:
            Number of documents loaded
        """
        if not self.enabled:
            return 0
        
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Knowledge directory not found: {directory}")
            return 0
        
        count = 0
        for game_dir in dir_path.iterdir():
            if not game_dir.is_dir():
                continue
            
            game = game_dir.name
            for category_dir in game_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                
                category = category_dir.name
                for file_path in category_dir.glob("*.txt"):
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        if content.strip():
                            self.add_document(
                                content=content,
                                game=game,
                                category=category,
                                metadata={"source_file": str(file_path)}
                            )
                            count += 1
                    except Exception as e:
                        logger.error(f"Failed to load {file_path}: {e}")
                
                for file_path in category_dir.glob("*.md"):
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        if content.strip():
                            self.add_document(
                                content=content,
                                game=game,
                                category=category,
                                metadata={"source_file": str(file_path)}
                            )
                            count += 1
                    except Exception as e:
                        logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {count} documents from {directory}")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if not self.enabled:
            return {"enabled": False, "reason": "RAG not available or disabled"}
        
        total = self.collection.count()
        
        # Get category distribution
        all_docs = self.collection.get(include=["metadatas"])
        categories = {}
        games = {}
        
        if all_docs["metadatas"]:
            for meta in all_docs["metadatas"]:
                cat = meta.get("category", "unknown")
                game = meta.get("game", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
                games[game] = games.get(game, 0) + 1
        
        return {
            "enabled": True,
            "total_documents": total,
            "categories": categories,
            "games": games,
            "embedding_model": self.kb_config.get("embedding_model")
        }
    
    def clear(self):
        """Clear all documents from the knowledge base."""
        if not self.enabled:
            return
        
        # Delete and recreate collection
        self.client.delete_collection("gaming_knowledge")
        self.collection = self.client.create_collection(
            name="gaming_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Knowledge base cleared")


def create_sample_knowledge_base(kb: GamingKnowledgeBase):
    """Create sample gaming knowledge for testing."""
    
    sample_docs = [
        # League of Legends - Builds
        GamingDocument(
            id="lol_jinx_build",
            content="""Jinx ADC Build Guide (Patch 14.12)
Core Build: Kraken Slayer -> Runaan's Hurricane -> Infinity Edge
Boots: Berserker's Greaves (Plated Steelcaps vs heavy AD, Mercury's Treads vs CC)
Situational: Lord Dominik's Regards (armor), Guardian Angel (safety), Mortal Reminder (healing), Bloodthirster (sustain)
Runes: Lethal Tempo (Presence of Mind, Legend: Bloodline, Coup de Grace) + Resolve (Conditioning, Overgrowth)
Summoner Spells: Flash + Heal (Cleanse vs heavy CC)
Skill Order: R > Q > W > E (Max Q first, then W, E last)
Early Game: Farm safely, use Q minigun for wave clear, rocket for poke
Mid Game: Look for picks with R, group for objectives
Late Game: Stay in backline, use rockets for siege, minigun for DPS""",
            game="League of Legends",
            category="build",
            metadata={"champion": "Jinx", "role": "ADC", "patch": "14.12"}
        ),
        
        # League of Legends - Strategy
        GamingDocument(
            id="lol_adc_strategy",
            content="""ADC Fundamentals - Playing from Behind
1. Wave Management: Freeze near your turret, deny CS, avoid overextending
2. Vision Control: Deep wards in enemy jungle, control wards in river bushes
3. Farming Priority: CS > Kills when behind. Every 15 CS ≈ 1 kill gold
4. Positioning: Stay at max range, use terrain, never facecheck bushes
5. Teamfight Target Priority: Closest enemy > Highest threat > Squishy targets
6. Comeback Mechanics: Look for picks with support, farm side lanes when safe
7. Itemization: Defensive items early (Vamp Scepter, Seeker's Armguard) if losing lane
8. Mental: Don't tilt, focus on CS, communicate with team""",
            game="League of Legends",
            category="strategy",
            metadata={"role": "ADC", "difficulty": "intermediate"}
        ),
        
        # Dota 2 - Build
        GamingDocument(
            id="dota2_pudge_build",
            content="""Pudge Position 4 Support Build (7.35)
Starting Items: Tango, Healing Salve, Clarity, Observer Ward, Sentry Ward
Early Game: Arcane Boots -> Blink Dagger -> Aether Lens
Core Items: Blink Dagger, Aether Lens, Force Staff, Glimmer Cape
Situational: Lotus Orb (vs single target), Aeon Disk (vs burst), Shiva's Guard (vs physical), 
Octarine Core (cooldown reduction), Refresher Orb (double hook)
Skill Build: Q (Meat Hook) > W (Rot) > E (Flesh Heap) > R (Dismember)
Max Q first, then E, W last. Take R at 6, 12, 18.
Talent Tree: Level 10: +20 Damage | Level 15: +1.5s Rot Slow | Level 20: +1 Hook Range | Level 25: Dismember Heals Allies
Playstyle: Hook -> Rot -> Dismember combo. Use Rot for wave clear and deny. 
Gank mid at level 2-3. Control runes. Stack camps for carry.""",
            game="Dota 2",
            category="build",
            metadata={"hero": "Pudge", "role": "Position 4", "patch": "7.35"}
        ),
        
        # Valorant - Strategy
        GamingDocument(
            id="val_jett_strategy",
            content="""Jett Guide - Valorant Competitive Play
Role: Duelist / Entry Fragger
Key Mechanics: Updraft + Dash (engage/disengage), Tailwind (reposition), Cloudburst (smoke), Blade Storm (ult)
Attack Side: 
- Entry: Dash in -> Updraft -> Shoot -> Dash out
- Operator Jett: Hold angles, use Updraft for off-angles
- Smoke for team entry, save dash for escape
Defense Side:
- Aggressive: Push early, use dash to retreat
- Passive: Hold angle, use updraft for verticality
- Retake: Smoke site, dash in, ult for multikill
Economy: Buy Operator when possible, save for full buy rounds
Maps: Best on maps with verticality (Bind, Haven, Ascent, Pearl)
Practice: Aim training (KovaaK's, Aim Lab), movement tech (super dash, updraft dash)""",
            game="Valorant",
            category="strategy",
            metadata={"agent": "Jett", "role": "Duelist"}
        ),
        
        # General Gaming - Mechanics
        GamingDocument(
            id="general_kiting",
            content="""Kiting / Orb Walking Fundamentals (Applies to LoL, Dota 2, etc.)
Definition: Attack -> Move -> Attack -> Move pattern to maximize damage while staying safe
Technique:
1. Issue attack command on target
2. Immediately issue move command away from target (or sideways)
3. Wait for attack animation to complete (attack windup)
4. Repeat
Key Concepts:
- Attack Windup: Time between attack command and projectile launch
- Attack Winddown: Time after projectile launches before you can move
- Animation Canceling: Move during winddown to cancel unnecessary animation
Practice Drills:
1. Practice tool: Attack dummy, move between each auto
2. Custom game: 1v1 vs bot, focus only on kiting
3. Target champions with slow projectiles (Jinx, Caitlyn, Drow Ranger)
Common Mistakes:
- Moving too early (cancels attack)
- Moving too late (wastes time)
- Not attacking at all (just running)
- Attacking wrong target (closest vs priority)""",
            game="General",
            category="mechanics",
            metadata={"applies_to": ["League of Legends", "Dota 2", "HoTS", "Smite"]}
        ),
        
        # Patch Analysis Example
        GamingDocument(
            id="lol_patch_14_12",
            content="""League of Legends Patch 14.12 Key Changes
Jungle Changes:
- Jungle XP reduced early, increased late
- Pet system reworked: pets now grant different bonuses
- Smite damage adjusted
Champion Buffs: 
- Jinx: Q minigun ramp up faster, R execute threshold increased
- Lee Sin: Q damage increased, W shield increased
- Ahri: Charm duration increased at max rank
Champion Nerfs:
- K'Sante: Base stats reduced, R cooldown increased
- Azir: Soldier damage reduced, shuffle cooldown increased
- Maokai: Sapling damage reduced, passive healing reduced
Item Changes:
- Kraken Slayer: True damage reduced
- Terminus: Stats adjusted
- New support items added
Meta Impact: 
- Scaling junglers stronger (Kayn, Master Yi)
- Early game junglers weaker (Lee Sin, Elise still viable but harder)
- ADC itemization shifts toward Runaan's Hurricane second item
- Mid lane control mages favored""",
            game="League of Legends",
            category="patch",
            metadata={"patch": "14.12", "type": "gameplay"}
        ),
    ]
    
    kb.add_documents_batch(sample_docs)
    logger.info("Sample knowledge base created")


def create_knowledge_base(config: Dict[str, Any]) -> GamingKnowledgeBase:
    """Factory function to create a GamingKnowledgeBase instance."""
    return GamingKnowledgeBase(config)