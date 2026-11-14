"""Text-to-speech workflow using the Mistral Voxtral TTS API.

Input: a text plus optional voice and format. The activity chunks long texts,
calls POST /v1/audio/speech per chunk, concatenates the MP3 frames, and returns
the audio as base64 in the execution result. A query handler exposes the audio
in slices so callers can page through it if the result exceeds their transport
limit.
"""

import base64
from typing import Optional

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import get_mistral_client

TTS_MODEL = "voxtral-mini-tts-3-0"
DEFAULT_VOICE = "voice_592"  # Marie, French female preset
DEFAULT_FORMAT = "mp3"
CHUNK_CHARS = 1500  # per-request text limit safety margin
MAX_RESULT_BYTES = 1_900_000  # stay under the 2MB payload limit


class TTSInput(BaseModel):
    text: str = Field(..., description="Texte à synthétiser en audio")
    voice_id: str = Field(DEFAULT_VOICE, description="Voix preset Voxtral, ex. voice_592 (Marie, français)")
    response_format: str = Field(DEFAULT_FORMAT, description="Format audio, ex. mp3")


class TTSOutput(BaseModel):
    audio_base64: str
    audio_bytes: int
    response_format: str
    voice_id: str
    model: str
    chunks: int


@workflows.activity()
async def generate_speech(params: TTSInput) -> TTSOutput:
    """Call the Voxtral TTS API; all network I/O lives in this activity."""
    client = get_mistral_client()
    text = params.text.strip()
    if not text:
        raise ValueError("Le texte à synthétiser est vide.")
    pieces = [text[i : i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
    audio = bytearray()
    for piece in pieces:
        response = client.audio.speech.complete(
            model=TTS_MODEL,
            input=piece,
            voice_id=params.voice_id,
            response_format=params.response_format,
        )
        audio.extend(base64.b64decode(response.audio_data))
    if len(audio) > MAX_RESULT_BYTES:
        raise ValueError(
            f"Audio trop volumineux ({len(audio)} octets, max {MAX_RESULT_BYTES}); raccourcissez le texte."
        )
    return TTSOutput(
        audio_base64=base64.b64encode(bytes(audio)).decode("ascii"),
        audio_bytes=len(audio),
        response_format=params.response_format,
        voice_id=params.voice_id,
        model=TTS_MODEL,
        chunks=len(pieces),
    )


@workflows.workflow.define(
    name="tts-generate",
    workflow_display_name="TTS Generate",
    workflow_description="Génère un fichier audio en Voxtral TTS à partir d'un texte.",
)
class TTSGenerateWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: TTSInput) -> TTSOutput:
        result = await generate_speech(input)
        self._audio_base64 = result.audio_base64
        self._audio_bytes = result.audio_bytes
        return result

    @workflows.workflow.query(name="audio_chunk")
    async def audio_chunk(
        self,
        offset: int = 0,
        length: int = 100_000,
    ) -> dict:
        """Return a base64 slice of the generated audio for paged retrieval."""
        data: Optional[str] = getattr(self, "_audio_base64", None)
        if not data:
            return {"available": False, "total_chars": 0, "offset": 0, "chunk": ""}
        return {
            "available": True,
            "total_chars": len(data),
            "offset": offset,
            "chunk": data[offset : offset + length],
        }
