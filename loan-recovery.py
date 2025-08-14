import argparse
import asyncio
import os
from datetime import datetime
import math
import calendar
from typing import Dict
from typing import Optional, List
from dotenv import load_dotenv
from loguru import logger
from typing import Set
import json
from fastapi import WebSocket
from collections import deque
import numpy as np
import os
from datetime import date 
import argparse
import asyncio
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.frames.frames import TranscriptionMessage, TranscriptionUpdateFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import TranscriptionMessage, TTSSpeakFrame, CancelFrame
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
from pipecat.transcriptions.language import Language
from pipecat.audio.filters.noisereduce_filter import NoisereduceFilter
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai_realtime_beta import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    OpenAIRealtimeBetaLLMService,
    SemanticTurnDetection,
    SessionProperties,
)
load_dotenv(override=True)


def load_instrcutions(path:str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

instruction_path = os.path.join(os.path.dirname(__file__), "info1.txt")
instruction_text = load_instrcutions(instruction_path)


class BorrowerRepo:
    """Very simple JSON-backed repo. Switch to SQLite/Redis later without touching the rest of code."""
    def __init__(self, path: str):
        self._path = path
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._db = json.load(f)
        else:
            self._db = {}

    async def get_by_id(self, borrower_id: str) -> Optional[dict]:
        # Simulate I/O (and yield control)
        await asyncio.sleep(0)
        return self._db.get(borrower_id)

# Init the repo
current_dir = os.getcwd()
path=os.path.join(current_dir, "borrowers.json")
BORROWER_REPO = BorrowerRepo(path = path)


BORROWER_STORE: Dict[str, dict] = {}
def natural_amount_phrase(amount: int, lang_hint: str = "hinglish") -> str:
    # super simple rules that sound natural enough
    # tweak buckets as you like
    if 900 <= amount < 1500:
        return "lagbhag hazaar" if lang_hint=="hinglish" else "around one thousand"
    if 1500 <= amount < 2500:
        return "dhai hazaar ke aas-paas" if lang_hint=="hinglish" else "around two thousand, maybe twenty-five hundred"
    if 2500 <= amount < 3500:
        return "karib teen hazaar" if lang_hint=="hinglish" else "around three thousand"
    if 3500 <= amount < 5500:
        return "saade chaar hazaar" if lang_hint=="hinglish" else "around forty-five hundred"
    if 5500 <= amount < 7500:
        return "saade paanch hazaar" if lang_hint=="hinglish" else "around fifty-five hundred"
    # fallback
    return f"karib {round(amount/1000)*1000} rupees" if lang_hint=="hinglish" else f"around {round(amount/1000)*1000} rupees"

def natural_date_phrase(iso_date: str, lang_hint: str = "hinglish") -> str:
    # "2025-08-05" -> "5 August" or "August 5th"
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    if lang_hint == "hinglish":
        return f"{dt.day} August" if dt.month == 8 else f"{dt.day} {calendar.month_name[dt.month]}"
    else:
        # English with ordinal
        day = dt.day
        suffix = "th" if 11<=day<=13 else {1:"st",2:"nd",3:"rd"}.get(day%10, "th")
        return f"{calendar.month_name[dt.month]} {day}{suffix}"

def make_save_promise_to_pay(borrower_store: Dict[str, dict]):
    async def save_promise_to_pay(params: FunctionCallParams):
        borrower_id = params.arguments["borrower_id"]
        ctx = borrower_store.get(borrower_id)

        if not ctx:
            # No borrower loaded for this ID; fail safely
            await params.result_callback({"status": "error", "reason": "unknown_borrower"})
            return

        amount_due = int(ctx["amount_due"])  # authoritative value from server context
        promised_amount = int(params.arguments["ptp_amount"])
        ptp_date = params.arguments["ptp_date"]
        delay_reason = params.arguments.get("delay_reason", "")
        negotiation_result = params.arguments.get("negotiation_result", "")

        # business rule: only accept if >= ₹900 (or add your % threshold internally)
        if promised_amount < 900:
            await params.result_callback({"status": "rejected", "reason": "amount_below_min"})
            return

        # save
        os.makedirs("ptp_logs", exist_ok=True)
        record = {
            "borrower_id": borrower_id,
            "loan_account_id": ctx.get("loan_account_id"),
            "ptp_date": ptp_date,
            "ptp_amount": promised_amount,
            "delay_reason": delay_reason,
            "negotiation_result": negotiation_result,
            "amount_due_at_ptp": amount_due,
        }
        with open(f"ptp_logs/{borrower_id}_{ptp_date}.txt", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        await params.result_callback({"status": "saved"})
    return save_promise_to_pay

def make_load_borrower_context(repo: BorrowerRepo,
                               borrower_store: Dict[str, dict],
                               context: OpenAILLMContext,
                               context_aggregator,
                               task: PipelineTask):

    async def load_borrower_context(params: FunctionCallParams):
        borrower_id = params.arguments["borrower_id"].strip()
        rec = await repo.get_by_id(borrower_id)

        if not rec:
            # speak a brief clarification and return not_found
            await params.result_callback({"status": "not_found"})
            return

        # Build humanized phrases
        amount_phrase_hi = natural_amount_phrase(int(rec["amount_due"]), "hinglish")
        amount_phrase_en = natural_amount_phrase(int(rec["amount_due"]), "english")
        due_date_hi      = natural_date_phrase(rec["emi_due_date"], "hinglish")
        due_date_en      = natural_date_phrase(rec["emi_due_date"], "english")

        # Save server-side store for other tools (e.g., PTP)
        borrower_store[borrower_id] = rec

        # Compose hidden system context
        extra_message = {
            "role": "system",
            "content": (
                "### BORROWER_CONTEXT (do not read aloud)\n"
                f"BorrowerName: {rec['borrower_name']}\n"
                f"LoanType: {rec['loan_type']}\n"
                f"AmountDueRaw: {rec['amount_due']}\n"
                f"AmountPhraseHinglish: {amount_phrase_hi}\n"
                f"AmountPhraseEnglish: {amount_phrase_en}\n"
                f"DueDateISO: {rec['emi_due_date']}\n"
                f"DueDateHinglish: {due_date_hi}\n"
                f"DueDateEnglish: {due_date_en}\n"
                f"BorrowerID: {rec['borrower_id']}\n"
                f"AccountID: {rec['loan_account_id']}\n"
                "\n"
                "### WHEN EXPLAINING DUE\n"
                "- If Hinglish, use AmountPhraseHinglish and DueDateHinglish\n"
                "- If English,  use AmountPhraseEnglish and  DueDateEnglish\n"
                "- Do not say exact rupees or ISO dates.\n"
            )
        }

        # Inject it into the model’s context live
        context.messages.append(extra_message)
        # await task.queue_frames([context_aggregator.user().get_context_frame()])

        await params.result_callback({"status": "loaded", "borrower_id": borrower_id})

    return load_borrower_context


load_borrower_context_schema = FunctionSchema(
    name="load_borrower_context",
    description=(
        "Given a borrower_id spoken by the user, fetch the borrower, "
        "push a hidden BORROWER_CONTEXT system message, and return status. "
        "Call this tool right after the user provides borrower_id."
    ),
    properties={
        "borrower_id": {"type": "string", "description": "The borrower id as provided by the user"}
    },
    required=["borrower_id"]
)


save_promise_to_pay_schema = FunctionSchema(
    name="save_promise_to_pay",
    description="Save promise to pay ONLY when amount is at least 20% of due amount and after negotiation attempts",
    properties={
        "borrower_id": {"type": "string", "description": "Borrower's ID for saving promise"},
        "ptp_date": {"type": "string", "description": "Date when borrower promised to pay (YYYY-MM-DD)"},
        "ptp_amount": {"type": "number", "description": "Amount promised (must be at least 20% of due amount)"},
        "delay_reason": {"type": "string", "description": "Reason for delay in payment"},
        "negotiation_result": {"type": "string", "description": "Brief note on negotiation outcome (e.g., 'Negotiated from 100 to 1200')"}
    },
    required=["borrower_id", "ptp_date", "ptp_amount", "negotiation_result"]
)





class TranscriptHandler:
    """Handles real-time transcript processing and output.

    Maintains a list of conversation messages and outputs them either to a log
    or to a file as they are received. Each message includes its timestamp and role.

    Attributes:
        messages: List of all processed transcript messages
        output_file: Optional path to file where transcript is saved. If None, outputs to log only.
    """

    def __init__(self, output_file: Optional[str] = None):
        """Initialize handler with optional file output.

        Args:
            output_file: Path to output file. If None, outputs to log only.
        """
        self.messages: List[TranscriptionMessage] = []
        self.output_file: Optional[str] = output_file
        logger.debug(
            f"TranscriptHandler initialized {'with output_file=' + output_file if output_file else 'with log output only'}"
        )

    async def save_message(self, message: TranscriptionMessage):
        """Save a single transcript message.

        Outputs the message to the log and optionally to a file.

        Args:
            message: The message to save
        """
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}{message.role}: {message.content}"

        # Always log the message
        logger.info(f"Transcript: {line}")

        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        # Optionally write to file
        if self.output_file:
            try:
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                logger.error(f"Error saving transcript message to file: {e}")

    async def on_transcript_update(
        self, processor: TranscriptProcessor, frame: TranscriptionUpdateFrame
    ):
        """Handle new transcript messages.

        Args:
            processor: The TranscriptProcessor that emitted the update
            frame: TranscriptionUpdateFrame containing new messages
        """
        logger.debug(f"Received transcript update with {len(frame.messages)} new messages")

        for msg in frame.messages:
            self.messages.append(msg)
            await self.save_message(msg)

async def run_bot(webrtc_connection: SmallWebRTCConnection, _: argparse.Namespace):
    logger.info(f"Starting bot")
    CALL_TIMEOUT_SECS = 240  # 4 minutes
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_filter = NoisereduceFilter(),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                        start_secs=0.20,   # react ~100 ms after speech onset
                        confidence=0.5,
                        stop_secs=0.4,
                        min_volume=0.6,
                    )),
        ),
    )
    


    session_properties = SessionProperties(
        input_audio_transcription=InputAudioTranscription(),
        # Set openai TurnDetection parameters. Not setting this at all will turn it
        # on by default
        turn_detection=SemanticTurnDetection(),
        # Or set to False to disable openai turn detection and use transport VAD
        # turn_detection=False,
        input_audio_noise_reduction=InputAudioNoiseReduction(type="near_field"),
        # tools=tools,
        instructions=f"{instruction_text}"
    )

    # Update STT to be more language-agnostic initially
    # stt = GroqSTTService(
    #     model="whisper-large-v3-turbo",
    #     api_key=os.getenv("GROQ_API_KEY"),
    #     language=Language.HI,  # Keep Hindi as base
    #     # prompt="Detect and transcribe in the original language spoken. Common terms: EMI, loan, payment, due, credit score, CIBIL, settlement. If Hindi/Hinglish, use Devanagari script. If English, use English script."
    # )

    
    api_key = os.getenv("OPENAI_API_KEY")
    stt = OpenAISTTService(
        api_key=api_key,
        model="gpt-4o-transcribe",
        prompt="Detect and transcribe in the original language spoken. Common terms: EMI, loan, payment, due, credit score, CIBIL, settlement. If Hindi/Hinglish, use Devanagari script. If English, use English script.",
    )

    # voice = "shimmer" if datetime.now().hour < 18 else "echo"
    tts = OpenAITTSService(api_key=api_key, voice="coral", model = "gpt-4o-mini-tts")


    # llm = OpenAILLMService(
    #     model = "gpt-4.1",
    #     api_key=api_key,
    #     temperature = 0.7
    # )


    llm = OpenAIRealtimeBetaLLMService(
        api_key=api_key,
        session_properties=session_properties,
        start_audio_paused=False,
    )

    # llm.register_function("save_promise_to_pay", save_promise_to_pay)
    

    tools = ToolsSchema(standard_tools=[
        load_borrower_context_schema,
        save_promise_to_pay_schema
    ])


    #Transcript handling to log the transcript file
    transcript = TranscriptProcessor()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_handler = TranscriptHandler(output_file=f"transcripts/session_{session_id}_{date.today()}.txt")

    messages = [
        {
            "role": "user",
            "content": "say hello"
            
        }
    ]

    context = OpenAILLMContext(messages, tools)

    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  
            # stt,
            # transcript.user(),
            context_aggregator.user(),
            llm,  
            transcript.user(),
            # tts,
            transport.output(),  
            transcript.assistant(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    llm.register_function("save_promise_to_pay", 
                          make_save_promise_to_pay(BORROWER_STORE)
                          )

    llm.register_function(
    "load_borrower_context",
    make_load_borrower_context(BORROWER_REPO, BORROWER_STORE, context, context_aggregator, task)
    )

    async def stop_session():
        logger.info(f"Stopping session in room")

        try:
            # Queue a cancel frame to stop the pipeline
            await asyncio.sleep(CALL_TIMEOUT_SECS)
            logger.info("Auto-ending call after 3 minutes")
            await task.queue_frame(TTSSpeakFrame("Thank you for your time. This call will now end."))
            await task.queue_frame(CancelFrame())
            # Cancel the task
            await task.cancel()
        except Exception as e:
            logger.error(f"Error stopping session: {e}")

    def get_greeting():
        hour = datetime.now().hour
        if hour < 12:
            return "Good morning"
        elif hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"



    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client csageonnected")



        # Kick off the conversation. to let LLM follow the system instruction
         # Dummy loan info fetched at connect time (can come from API/database too)
        # borrower_context = {
        #     "borrower_id":"LKU7878",
        #     "borrower_name": "Rahul Sharma",
        #     "amount_due": 4578,
        #     "last_payment_date": "2025-07-10",
        #     "emi_due_date": "2025-08-05",
        #     "loan_type": "Personal Loan",
        #     "overdue_days": 30,
        #     "loan_account_id": "LNX-8392937"
        # }

        # borrower_id = "LKU7878" #borrower_id or os.getenv("DEMO_BORROWER_ID", "LKU7878")
        # borrower_context = await BORROWER_REPO.get_by_id(borrower_id)
        # if not borrower_context:
        #     logger.warning(f"Unknown borrower_id={borrower_id}; using fallback")
        #     borrower_context = {
        #         "borrower_id":"LKU7878",
        #         "borrower_name": "Rahul Sharma",
        #         "amount_due": 4578,
        #         "last_payment_date": "2025-07-10",
        #         "emi_due_date": "2025-08-05",
        #         "loan_type": "Personal Loan",
        #         "overdue_days": 30,
        #         "loan_account_id": "LNX-8392937"
        #     }

        # amount_phrase_hinglish = natural_amount_phrase(borrower_context["amount_due"], "hinglish")
        # amount_phrase_english  = natural_amount_phrase(borrower_context["amount_due"], "english")
        # due_date_hinglish      = natural_date_phrase(borrower_context["emi_due_date"], "hinglish")
        # due_date_english       = natural_date_phrase(borrower_context["emi_due_date"], "english")

        # # Add calculated thresholds
        # min_acceptable = int(borrower_context['amount_due'] * 0.25)
        # target_amount = int(borrower_context['amount_due'] * 0.50)
        # good_amount = int(borrower_context['amount_due'] * 0.75)


        #  # Append this info as a user/system context message
        # extra_message = {
        #     "role": "system",
        #     "content": (
        #         "### BORROWER_CONTEXT (do not read aloud)\n"
        #         f"BorrowerName: {borrower_context['borrower_name']}\n"
        #         f"LoanType: {borrower_context['loan_type']}\n"
        #         f"AmountDueRaw: {borrower_context['amount_due']}\n"
        #         f"AmountPhraseHinglish: {amount_phrase_hinglish}\n"
        #         f"AmountPhraseEnglish: {amount_phrase_english}\n"
        #         f"DueDateISO: {borrower_context['emi_due_date']}\n"
        #         f"DueDateHinglish: {due_date_hinglish}\n"
        #         f"DueDateEnglish: {due_date_english}\n"
        #         f"BorrowerID: {borrower_context['borrower_id']}\n"
        #         f"AccountID: {borrower_context['loan_account_id']}\n"
        #         "\n"
        #         "### FIRST_TURN\n"
        #         "- Only greet by name + intro, then pause and wait for reply.\n"
        #         "- Hinglish greeting: 'Hello {BorrowerName} ji, main Raj bol raha hun XYZ Company se'\n"
        #         "- English greeting:  'Hi {BorrowerName}, this is Raj from XYZ Company'\n"
        #         "\n"
        #         "### WHEN EXPLAINING DUE\n"
        #         "- If Hinglish, use AmountPhraseHinglish and DueDateHinglish\n"
        #         "- If English,  use AmountPhraseEnglish and  DueDateEnglish\n"
        #         "- Do not say exact rupees or ISO dates.\n"
        #     )
        # }
        # BORROWER_STORE[borrower_context["borrower_id"]] = borrower_context
        # context.messages.append(extra_message)
        first_turn = {
            "role": "system",
            "content": (
                "### FIRST_TURN\n"
                "- Greet the caller briefly.\n"
                "- Ask for their borrower ID clearly (e.g., 'Kripya apna borrower ID batayein').\n"
                "- Wait for the user to respond. Do NOT assume or fabricate the ID.\n"
                "- As soon as the user provides the ID, CALL the tool `load_borrower_context` with it.\n"
                "- After the tool returns status=loaded, continue the conversation using BORROWER_CONTEXT.\n"
            )
        }
        context.messages.append(first_turn)
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        asyncio.create_task(stop_session())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")

    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        # for msg in frame.messages:
        #     if isinstance(msg, TranscriptionMessage):
        #         timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
        #         line = f"{timestamp}{msg.role}: {msg.content}"
        #         logger.info(f"Transcript: {line}")
        await transcript_handler.on_transcript_update(processor, frame)

    await PipelineRunner(handle_sigint=False).run(task)



if __name__ == "__main__":
    from run import main
    main()
