from app.extraction.extractor import Extractor


def test_character_extraction_from_story():
    story = "abcdefghijklmnopqrstuvwxyz"
    positions = [0, 2, 4, 6, 8]
    extracted = Extractor.extract(story, positions)
    assert extracted == "acegi"
