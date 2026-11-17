"""YouTube の配信URLから動画IDを取り出すテスト"""
import pytest

from backend.sources.video_id import extract_youtube_id

ID = "AbCdEfGh-_1"


@pytest.mark.parametrize("text", [
    ID,
    f"https://www.youtube.com/watch?v={ID}",
    f"https://www.youtube.com/watch?v={ID}&t=120s",
    f"https://www.youtube.com/watch?feature=share&v={ID}",
    f"https://youtu.be/{ID}?si=xyz",
    f"https://www.youtube.com/live/{ID}?feature=shared",
    f"https://youtube.com/shorts/{ID}",
    f"https://www.youtube.com/embed/{ID}",
    f"https://studio.youtube.com/video/{ID}/livestreaming",
    f"  https://m.youtube.com/watch?v={ID}  ",
])
def test_extracts_id(text):
    assert extract_youtube_id(text) == ID


@pytest.mark.parametrize("text", ["mock", "local", "my_twitch_channel", "", "https://twitcasting.tv/user"])
def test_leaves_non_youtube_input_alone(text):
    assert extract_youtube_id(text) == text.strip()
