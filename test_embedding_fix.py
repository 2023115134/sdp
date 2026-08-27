#!/usr/bin/env python3
"""
Quick test to verify the embedding optimization works.
Tests the scenario that previously failed.
"""

from app.crypto.mapping import CharacterMap
from app.crypto.position_generator import PositionGenerator
from app.llm.embedder import EmbedderLLM
from app.llm.generator import LLMGenerator

def test_embedding():
    """Test embedding with the same scenario that previously failed."""
    
    print("Testing embedding with optimized parameters...")
    print("=" * 70)
    
    # Use simple inputs
    secret = "hi"
    topic = "a boy is sitting on a bench"
    
    print(f"Secret: {secret}")
    print(f"Topic: {topic}")
    print()
    
    # Map and generate positions
    character_map = CharacterMap()
    mapped = character_map.encode(secret)
    print(f"Mapped: {mapped}")
    
    generator = PositionGenerator(offset_do=32, max_story_length=1000)
    positions = generator.generate_for_message(
        key="test-key",
        message_length=len(mapped),
    )
    print(f"Positions: {positions}")
    print()
    
    # Create embedder with optimized parameters
    llm_gen = LLMGenerator()
    embedder = EmbedderLLM(
        llm_generator=llm_gen,
        character_map=character_map,
    )
    
    print("Starting embedding with optimized parameters...")
    print("(max_retries=8, max_attempts=75000)")
    print()
    
    try:
        result = embedder.embed(
            topic=topic,
            characters=mapped,
            positions=positions,
            temperature=0.70,
            top_k=40,
            max_new_tokens=8,
            max_attempts=75000,
            max_retries=8,
        )
        
        print("✅ EMBEDDING SUCCESSFUL!")
        print()
        print("Generated story:")
        print(result.story)
        print()
        print(f"Embedded characters: {result.embedded_characters}")
        print(f"Positions used: {result.positions}")
        print(f"Total attempts: {result.attempts}")
        
        return True
        
    except RuntimeError as e:
        print(f"❌ EMBEDDING FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_embedding()
    exit(0 if success else 1)
