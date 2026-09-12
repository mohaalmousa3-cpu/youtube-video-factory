"""CLI entry point — mirrors youtube-intelligence-engine's argparse style."""
import argparse

from src.database.db import init_db
from src.utils.logging import setup_logging


def _check(logger, label: str, check) -> bool:
    try:
        if check():
            logger.info("%s: OK", label)
            return True
        logger.error("%s: FAILED", label)
        return False
    except Exception as exc:
        logger.error("%s: FAILED (%s)", label, exc)
        return False


def cmd_health(_args: argparse.Namespace) -> int:
    logger = setup_logging()

    from src.providers.llm_groq import GroqProvider
    from src.providers.llm_tokenrouter import TokenRouterProvider
    from src.providers.tts_kokoro import KokoroProvider
    from src.providers.image_qwen import QwenImageProvider
    from src.providers import lipsync_rhubarb, image_upscale
    from src.render import manim_render, ffmpeg_render

    results = [
        _check(logger, "Groq (LLM)", GroqProvider().health_check),
        _check(logger, "TokenRouter (LLM fallback)", TokenRouterProvider().health_check),
        _check(logger, "Kokoro (TTS)", KokoroProvider().health_check),
        _check(logger, "Qwen-Image (backgrounds/character)", QwenImageProvider().health_check),
        _check(logger, "Rhubarb (lip sync)", lipsync_rhubarb.health_check),
        _check(logger, "Real-ESRGAN (upscale)", image_upscale.health_check),
        _check(logger, "Manim (supporting animation)", manim_render.health_check),
        _check(logger, "FFmpeg (render)", ffmpeg_render.health_check),
    ]

    ok = all(results)
    logger.info("Health check %s", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_init_db(_args: argparse.Namespace) -> int:
    logger = setup_logging()
    init_db()
    logger.info("Database initialized.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="video-factory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Check all providers are reachable/working").set_defaults(func=cmd_health)
    sub.add_parser("init-db", help="Create the jobs database").set_defaults(func=cmd_init_db)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
