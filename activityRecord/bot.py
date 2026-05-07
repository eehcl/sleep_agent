import os
import re
import json
import sqlite3
import asyncio
import logging
import urllib.parse
from datetime import datetime, timedelta
import ollama
import requests
from dotenv import load_dotenv
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ConversationHandler,
    ContextTypes,
)

GEMMA_MODEL = "gemma4:e2b"

# 환경 변수 로드
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleep_agent.db")


def init_db() -> None:
    """sessions 테이블이 없으면 생성하고, 빠진 컬럼은 마이그레이션한다."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                fatigue TEXT,
                stress TEXT,
                caffeine TEXT,
                mood TEXT,
                wake_time TEXT,
                target_time TEXT,
                sleep_latency TEXT,
                youtube_query TEXT,
                youtube_title TEXT,
                youtube_link TEXT,
                feedback TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "sleep_latency" not in existing:
            conn.execute("ALTER TABLE sessions ADD COLUMN sleep_latency TEXT")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                auto_volume INTEGER NOT NULL DEFAULT 0
            )
            """
        )


def _get_auto_volume_sync(user_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT auto_volume FROM user_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return bool(row[0]) if row else False


async def get_auto_volume(user_id: int) -> bool:
    return await asyncio.to_thread(_get_auto_volume_sync, user_id)


def _set_auto_volume_sync(user_id: int, enabled: bool) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO user_settings (user_id, auto_volume) VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET auto_volume = excluded.auto_volume
            """,
            (user_id, 1 if enabled else 0),
        )


async def set_auto_volume(user_id: int, enabled: bool) -> None:
    await asyncio.to_thread(_set_auto_volume_sync, user_id, enabled)


def _save_session_sync(user_id: int, data: dict) -> int:
    now = datetime.now()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO sessions (
                user_id, date, fatigue, stress, caffeine, mood,
                wake_time, target_time, sleep_latency, youtube_query,
                youtube_title, youtube_link, feedback, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                now.strftime("%Y-%m-%d"),
                data.get("fatigue"),
                data.get("stress"),
                data.get("caffeine"),
                data.get("mood"),
                data.get("wake_time"),
                data.get("target_time"),
                data.get("sleep_latency"),
                data.get("youtube_query"),
                data.get("youtube_title"),
                data.get("youtube_link"),
                None,
                now.isoformat(timespec="seconds"),
            ),
        )
        return cur.lastrowid


async def save_session(user_id: int, data: dict) -> int:
    return await asyncio.to_thread(_save_session_sync, user_id, data)


def _update_feedback_sync(session_id: int, feedback: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET feedback = ? WHERE id = ?",
            (feedback, session_id),
        )


async def update_feedback(session_id: int, feedback: str) -> None:
    await asyncio.to_thread(_update_feedback_sync, session_id, feedback)


def _get_recent_sessions_sync(user_id: int, limit: int) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT date, fatigue, stress, mood, wake_time,
                   youtube_title, youtube_link, feedback
            FROM sessions
            WHERE user_id = ? AND youtube_title IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


async def get_recent_sessions(user_id: int, limit: int = 3) -> list[dict]:
    return await asyncio.to_thread(_get_recent_sessions_sync, user_id, limit)


def _get_sessions_within_days_sync(user_id: int, days: int) -> list[dict]:
    cutoff_str = (datetime.now().date() - timedelta(days=days - 1)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT date, fatigue, stress, caffeine, mood,
                   wake_time, target_time, sleep_latency,
                   youtube_query, youtube_title, youtube_link, feedback
            FROM sessions
            WHERE user_id = ? AND date >= ?
            ORDER BY id ASC
            """,
            (user_id, cutoff_str),
        ).fetchall()
        return [dict(r) for r in rows]


async def get_sessions_within_days(user_id: int, days: int = 7) -> list[dict]:
    return await asyncio.to_thread(_get_sessions_within_days_sync, user_id, days)


def _parse_hhmm(text: str | None) -> int | None:
    """'HH:MM' 문자열을 자정 기준 분으로 변환. 잘못된 형식이면 None."""
    if not text:
        return None
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", text)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if 0 <= hour < 24 and 0 <= minute < 60:
        return hour * 60 + minute
    return None


def _planned_sleep_minutes(target_time: str | None, wake_time: str | None) -> int | None:
    """목표 취침 시간 → 기상 시간까지 분(자정 넘기면 +24h 보정)."""
    t = _parse_hhmm(target_time)
    w = _parse_hhmm(wake_time)
    if t is None or w is None:
        return None
    return (w - t) % (24 * 60)


def _format_minutes(total: int | None) -> str:
    if total is None:
        return "—"
    h, m = divmod(total, 60)
    if h and m:
        return f"{h}시간 {m}분"
    if h:
        return f"{h}시간"
    return f"{m}분"


def _parse_sleep_latency_minutes(text: str | None) -> int | None:
    """'15', '15분', '약 20분' 등에서 정수 분만 추출. '모름'/공백이면 None."""
    if not text:
        return None
    m = re.search(r"\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


FEEDBACK_LABELS = {"helpful": "도움됨", "not_helpful": "별로였음"}

# 대화 단계 정의
FATIGUE, STRESS, CAFFEINE, MOOD, WAKE_TIME, TARGET_TIME, SLEEP_LATENCY = range(7)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """대화 시작 및 인사"""
    await update.message.reply_text(
        "안녕하세요! 당신의 꿀잠을 도와주는 Sleep Agent입니다. 😴\n"
        "취침 전 현재 컨디션을 체크해 볼까요?\n\n"
        "먼저, 현재 **피곤함 정도**를 선택해 주세요. (1: 아주 쌩쌩함 ~ 5: 매우 피곤함)",
        reply_markup=ReplyKeyboardMarkup(
            [["1", "2", "3", "4", "5"]], one_time_keyboard=True
        ),
    )
    return FATIGUE

async def get_fatigue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """피곤함 입력 및 스트레스 질문"""
    context.user_data["fatigue"] = update.message.text
    await update.message.reply_text(
        "좋습니다. 그럼 현재 **스트레스 정도**는 어느 정도인가요? (1: 평온함 ~ 5: 매우 스트레스)",
        reply_markup=ReplyKeyboardMarkup(
            [["1", "2", "3", "4", "5"]], one_time_keyboard=True
        ),
    )
    return STRESS

async def get_stress(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """스트레스 입력 및 카페인 질문"""
    context.user_data["stress"] = update.message.text
    await update.message.reply_text(
        "오늘 **카페인**은 무엇을 몇 시쯤 드셨나요? (예: 오후 2시에 아아 한 잔, 없음 등)",
        reply_markup=ReplyKeyboardRemove(),
    )
    return CAFFEINE

async def get_caffeine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """카페인 입력 및 기분 질문"""
    context.user_data["caffeine"] = update.message.text
    await update.message.reply_text(
        "현재 **기분 상태**는 어떠신가요?",
        reply_markup=ReplyKeyboardMarkup(
            [["행복", "우울", "불안", "평온", "설렘"]], one_time_keyboard=True
        ),
    )
    return MOOD

async def get_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """기분 입력 및 내일 기상 시간 질문"""
    context.user_data["mood"] = update.message.text
    await update.message.reply_text(
        "**내일 기상 시간**을 알려주세요. (예: 07:00)",
        reply_markup=ReplyKeyboardRemove(),
    )
    return WAKE_TIME

async def get_wake_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """기상 시간 입력 및 목표 취침 시간 질문"""
    context.user_data["wake_time"] = update.message.text
    await update.message.reply_text(
        "마지막으로, **목표 취침 시간**을 알려주세요. (예: 23:30)",
        reply_markup=ReplyKeyboardRemove(),
    )
    return TARGET_TIME

def _format_history(history: list[dict]) -> str:
    """이전 세션 목록을 Gemma 프롬프트에 끼울 자연어로 변환."""
    if not history:
        return "이전 기록 없음 (이번이 첫 추천입니다)."

    lines = []
    for row in history:
        title = (row.get("youtube_title") or "").strip()
        if len(title) > 50:
            title = title[:50] + "…"
        feedback = row.get("feedback")
        feedback_text = FEEDBACK_LABELS.get(feedback, "피드백 미응답")
        lines.append(
            f"- {row['date']}: 피곤함 {row.get('fatigue')}, 스트레스 {row.get('stress')}, "
            f"기분 {row.get('mood')}, '{title}' 영상 시청 (피드백: {feedback_text})"
        )
    return "\n".join(lines)


async def get_recommendation_from_gemma(user_id: int, user_state: dict) -> str:
    """Gemma 모델에 사용자 상태와 이전 기록을 전달하고 수면 콘텐츠 추천을 받는다."""
    history = await get_recent_sessions(user_id, limit=3)
    history_block = _format_history(history)
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    prompt = (
        "당신은 사용자의 컨디션을 분석해 수면에 도움이 되는 콘텐츠(예: ASMR, 명상 가이드, 백색소음, 잔잔한 음악, 수면 동화 등)를 추천하는 전문가입니다.\n\n"
        "이 사용자의 최근 수면 콘텐츠 기록입니다 (최근 것부터):\n"
        f"{history_block}\n\n"
        f"현재 시각: {current_time_str}\n\n"
        "다음은 사용자의 오늘 현재 상태입니다:\n"
        f"- 피곤함 정도(1~5): {user_state['fatigue']}\n"
        f"- 스트레스 정도(1~5): {user_state['stress']}\n"
        f"- 오늘 카페인 섭취: {user_state['caffeine']}\n"
        f"- 현재 기분: {user_state['mood']}\n"
        f"- 내일 기상 시간: {user_state['wake_time']}\n"
        f"- 목표 취침 시간: {user_state['target_time']}\n\n"
        "사용자가 입력한 카페인 정보를 분석할 때, 특정 브랜드(스타벅스, 일리, 네스프레소 등)나 음료 종류가 언급되면 너의 지식을 바탕으로 **예상 카페인 함량(mg)**을 먼저 계산해 봐. "
        "그리고 그 함량이 현재 사용자의 수면에 어떤 영향을 미칠지(예: 카페인 반감기 고려 등)를 설명에 포함해서 아주 전문적으로 대답해 줘.\n"
        "사용자가 입력한 카페인 섭취 시간과 현재 시간을 비교하여, 카페인이 수면에 미치는 영향을 과학적으로 분석해서 답변에 포함해 줘.\n\n"
        "이전 기록에서 '도움됨' 피드백을 받은 콘텐츠와 비슷한 결을 선호하고, '별로였음' 피드백을 받은 것과는 다른 방향으로 추천하세요. "
        "최근에 본 영상과 똑같은 영상을 다시 추천하지 마세요.\n\n"
        "다음 세 가지를 함께 알려주세요:\n"
        "1) 이 상태에 가장 잘 어울리는 수면 콘텐츠 추천과 그 이유 (이전 기록을 어떻게 반영했는지 포함)\n"
        "2) 사용자의 기상 시간을 고려했을 때 권장하는 취침 시간과 총 수면 시간\n"
        "3) 이 사용자에게 딱 맞는 유튜브 검색어 (한 문장)\n\n"
        "응답의 마지막 줄은 반드시 다음 형식으로만 작성하세요 (다른 말은 붙이지 말 것):\n"
        "유튜브 검색어: <검색어 한 문장>"
    )

    client = ollama.AsyncClient()
    response = await client.chat(
        model=GEMMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def extract_youtube_query(text: str) -> str | None:
    """Gemma 응답에서 '유튜브 검색어:' 라인을 찾아 검색어를 추출한다."""
    match = re.search(r"유튜브\s*검색어\s*[:：]\s*(.+)", text)
    if not match:
        return None
    query = match.group(1).strip()
    return query.strip("\"'`*<> ") or None


_YOUTUBE_INITIAL_DATA_PATTERNS = (
    re.compile(r"var ytInitialData\s*=\s*(\{.+?\});\s*</script>", re.S),
    re.compile(r'ytInitialData"\]\s*=\s*(\{.+?\});\s*</script>', re.S),
)


def _find_first_video_renderer(node) -> dict | None:
    """ytInitialData 트리에서 가장 먼저 나오는 videoRenderer를 깊이우선으로 찾는다."""
    if isinstance(node, dict):
        vr = node.get("videoRenderer")
        if isinstance(vr, dict):
            video_id = vr.get("videoId")
            title_runs = (vr.get("title") or {}).get("runs") or []
            title = title_runs[0].get("text") if title_runs else None
            if video_id and title:
                return {
                    "title": title,
                    "link": f"https://www.youtube.com/watch?v={video_id}",
                }
        for value in node.values():
            found = _find_first_video_renderer(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_first_video_renderer(item)
            if found:
                return found
    return None


def _search_youtube_top_sync(query: str) -> dict | None:
    """YouTube 검색 결과 페이지를 직접 가져와 최상단 영상 1개를 추출한다."""
    url = "https://www.youtube.com/results?" + urllib.parse.urlencode(
        {"search_query": query, "hl": "ko", "persist_hl": "1"}
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko,en;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    for pattern in _YOUTUBE_INITIAL_DATA_PATTERNS:
        match = pattern.search(resp.text)
        if not match:
            continue
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        return _find_first_video_renderer(data)
    return None


async def search_top_youtube_video(query: str) -> dict | None:
    """검색어로 유튜브 최상단 영상 1개의 제목과 링크를 가져온다."""
    return await asyncio.to_thread(_search_youtube_top_sync, query)


def _wearable_only_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(
            "애플워치/갤럭시워치 데이터 연동하기",
            callback_data="link_wearable",
        )]]
    )


def _feedback_keyboard(session_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "👍 도움이 됐어요",
                    callback_data=f"feedback:{session_id}:helpful",
                ),
                InlineKeyboardButton(
                    "👎 별로였어요",
                    callback_data=f"feedback:{session_id}:not_helpful",
                ),
            ],
            [
                InlineKeyboardButton(
                    "애플워치/갤럭시워치 데이터 연동하기",
                    callback_data="link_wearable",
                )
            ],
        ]
    )


async def get_target_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """목표 취침 시간 입력 후 어젯밤 잠드는 데 걸린 시간을 묻는다."""
    context.user_data["target_time"] = update.message.text
    await update.message.reply_text(
        "한 가지만 더요! **어젯밤 잠드는 데 걸린 시간**은 약 몇 분이었나요? "
        "(숫자만 입력, 예: 15. 모르면 '모름' 입력)",
        reply_markup=ReplyKeyboardRemove(),
    )
    return SLEEP_LATENCY


async def get_sleep_latency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """최종 입력 데이터 요약 및 Gemma 추천 전송"""
    context.user_data["sleep_latency"] = update.message.text
    user_id = update.effective_user.id

    summary = (
        "✅ 입력을 완료했습니다!\n\n"
        f"- 피곤함: {context.user_data['fatigue']}\n"
        f"- 스트레스: {context.user_data['stress']}\n"
        f"- 카페인: {context.user_data['caffeine']}\n"
        f"- 기분: {context.user_data['mood']}\n"
        f"- 내일 기상 시간: {context.user_data['wake_time']}\n"
        f"- 목표 취침 시간: {context.user_data['target_time']}\n"
        f"- 어젯밤 잠들기까지: {context.user_data['sleep_latency']}\n\n"
        "이제 이 데이터를 바탕으로 Gemma가 최적의 수면 콘텐츠를 분석 중입니다... 🧠"
    )
    await update.message.reply_text(summary)

    try:
        recommendation = await get_recommendation_from_gemma(user_id, context.user_data)
    except Exception as e:
        logger.exception("Gemma 추천 요청 중 오류 발생")
        await update.message.reply_text(
            f"⚠️ 추천을 가져오는 중 문제가 발생했습니다: {e}",
            reply_markup=_wearable_only_keyboard(),
        )
        return ConversationHandler.END

    query = extract_youtube_query(recommendation)
    display_text = re.sub(
        r"\n?유튜브\s*검색어\s*[:：].*$", "", recommendation, flags=re.S
    ).strip() or recommendation

    if await get_auto_volume(user_id):
        display_text = (
            "🔊 사용자가 잠들면 소리를 서서히 줄여주는 모드가 활성화되었습니다.\n\n"
            + display_text
        )

    await update.message.reply_text(f"🌙 Gemma의 수면 콘텐츠 추천\n\n{display_text}")

    video: dict | None = None
    youtube_error: Exception | None = None
    if query:
        try:
            video = await search_top_youtube_video(query)
        except Exception as e:
            youtube_error = e
            logger.exception("유튜브 검색 중 오류 발생")

    session_id = await save_session(
        user_id,
        {
            **context.user_data,
            "youtube_query": query,
            "youtube_title": video["title"] if video else None,
            "youtube_link": video["link"] if video else None,
        },
    )

    if not query:
        await update.message.reply_text(
            "⚠️ 추천 영상 검색어를 추출하지 못했습니다.",
            reply_markup=_wearable_only_keyboard(),
        )
        return ConversationHandler.END

    if youtube_error is not None:
        await update.message.reply_text(
            f"🔎 검색어: {query}\n⚠️ 유튜브 검색 중 문제가 발생했습니다: {youtube_error}",
            reply_markup=_wearable_only_keyboard(),
        )
        return ConversationHandler.END

    if not video:
        await update.message.reply_text(
            f"🔎 검색어: {query}\n검색 결과가 없습니다.",
            reply_markup=_wearable_only_keyboard(),
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "🎬 추천 영상\n\n"
        f"🔎 검색어: {query}\n"
        f"제목: {video['title']}\n"
        f"링크: {video['link']}\n\n"
        "이 영상이 수면에 도움이 되었나요? 아래 버튼으로 알려주세요.",
        reply_markup=_feedback_keyboard(session_id),
    )

    return ConversationHandler.END


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """수면 피드백 버튼 콜백을 처리한다."""
    query = update.callback_query
    await query.answer()

    parts = (query.data or "").split(":")
    if len(parts) != 3 or parts[0] != "feedback":
        return

    try:
        session_id = int(parts[1])
    except ValueError:
        return
    feedback = parts[2]
    if feedback not in FEEDBACK_LABELS:
        return

    try:
        await update_feedback(session_id, feedback)
    except Exception:
        logger.exception("피드백 저장 중 오류 발생")
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("⚠️ 피드백 저장 중 문제가 발생했습니다.")
        return

    label = FEEDBACK_LABELS[feedback]
    await query.edit_message_reply_markup(reply_markup=_wearable_only_keyboard())
    await query.message.reply_text(
        f"피드백 저장 완료 ✅ ('{label}'). 다음 추천에 반영할게요!"
    )

def _summarize_sessions(sessions: list[dict]) -> dict:
    """세션 리스트에서 평균/총합 등의 통계를 계산."""
    sleep_durations = [
        _planned_sleep_minutes(s.get("target_time"), s.get("wake_time"))
        for s in sessions
    ]
    sleep_durations = [d for d in sleep_durations if d is not None]
    latencies = [
        _parse_sleep_latency_minutes(s.get("sleep_latency"))
        for s in sessions
    ]
    latencies = [v for v in latencies if v is not None]

    helpful = sum(1 for s in sessions if s.get("feedback") == "helpful")
    not_helpful = sum(1 for s in sessions if s.get("feedback") == "not_helpful")

    return {
        "count": len(sessions),
        "avg_sleep": int(sum(sleep_durations) / len(sleep_durations)) if sleep_durations else None,
        "total_sleep": sum(sleep_durations) if sleep_durations else None,
        "avg_latency": int(sum(latencies) / len(latencies)) if latencies else None,
        "helpful": helpful,
        "not_helpful": not_helpful,
    }


def _format_report_lines(sessions: list[dict]) -> str:
    """일자별 리포트 라인을 생성."""
    lines = []
    for s in sessions:
        sleep = _format_minutes(_planned_sleep_minutes(s.get("target_time"), s.get("wake_time")))
        latency_min = _parse_sleep_latency_minutes(s.get("sleep_latency"))
        latency = _format_minutes(latency_min) if latency_min is not None else "—"
        title = (s.get("youtube_title") or "—").strip()
        if len(title) > 45:
            title = title[:45] + "…"
        feedback = FEEDBACK_LABELS.get(s.get("feedback"), "미응답")
        lines.append(
            f"• {s['date']} | 수면 {sleep} | 잠들기 {latency} | 영상: {title} | 피드백: {feedback}"
        )
    return "\n".join(lines)


async def get_pattern_analysis_from_gemma(sessions: list[dict]) -> str:
    """Gemma에 최근 세션을 전달해 수면 패턴 분석/개선 방향을 받는다."""
    rows = []
    for s in sessions:
        sleep = _format_minutes(_planned_sleep_minutes(s.get("target_time"), s.get("wake_time")))
        latency_min = _parse_sleep_latency_minutes(s.get("sleep_latency"))
        latency = f"{latency_min}분" if latency_min is not None else "기록 없음"
        feedback = FEEDBACK_LABELS.get(s.get("feedback"), "미응답")
        title = (s.get("youtube_title") or "없음").strip()
        rows.append(
            f"- {s['date']}: 피곤함 {s.get('fatigue')}, 스트레스 {s.get('stress')}, "
            f"카페인 {s.get('caffeine')}, 기분 {s.get('mood')}, "
            f"수면 {sleep}, 잠들기 {latency}, 영상 '{title}', 피드백 {feedback}"
        )
    data_block = "\n".join(rows)

    prompt = (
        "당신은 수면 코치입니다. 아래는 한 사용자의 최근 수면 기록입니다.\n\n"
        f"{data_block}\n\n"
        "최근 이 사용자의 수면 패턴을 분석해서 개선 방향을 알려주세요. "
        "구체적으로 다음을 다뤄 주세요:\n"
        "1) 눈에 띄는 패턴 (수면 시간 변동성, 카페인-수면 연관, 기분-수면 연관, 영상 효과 등)\n"
        "2) 개선이 필요한 지점 2~3가지\n"
        "3) 내일 밤부터 바로 실천할 수 있는 행동 제안 3가지\n\n"
        "한국어로 간결한 불릿 형태로 답해 주세요."
    )

    client = ollama.AsyncClient()
    response = await client.chat(
        model=GEMMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/report — 최근 7일 수면 리포트와 Gemma 패턴 분석을 전송."""
    user_id = update.effective_user.id
    sessions = await get_sessions_within_days(user_id, days=7)

    if not sessions:
        await update.message.reply_text(
            "📊 최근 7일간 기록이 아직 없어요.\n/start로 첫 세션을 시작해 보세요!"
        )
        return

    stats = _summarize_sessions(sessions)
    header = (
        "📊 최근 7일 수면 리포트\n\n"
        f"- 기록된 세션 수: {stats['count']}회\n"
        f"- 누적 수면 시간(계획 기준): {_format_minutes(stats['total_sleep'])}\n"
        f"- 평균 수면 시간: {_format_minutes(stats['avg_sleep'])}\n"
        f"- 평균 잠들기까지: {_format_minutes(stats['avg_latency'])}\n"
        f"- 영상 피드백: 👍 {stats['helpful']} / 👎 {stats['not_helpful']}\n"
    )
    await update.message.reply_text(header)

    body = "🗓 일자별 기록\n\n" + _format_report_lines(sessions)
    # 텔레그램 메시지 길이 제한(4096) 보호
    for chunk_start in range(0, len(body), 3500):
        await update.message.reply_text(body[chunk_start : chunk_start + 3500])

    if stats["count"] < 3:
        await update.message.reply_text(
            "💡 패턴 분석은 3회 이상 기록이 쌓이면 제공됩니다. 며칠만 더 기록해 주세요!"
        )
        return

    await update.message.reply_text("🧠 Gemma가 수면 패턴을 분석하고 있어요...")
    try:
        analysis = await get_pattern_analysis_from_gemma(sessions)
    except Exception as e:
        logger.exception("Gemma 패턴 분석 중 오류 발생")
        await update.message.reply_text(f"⚠️ 패턴 분석 중 문제가 발생했습니다: {e}")
        return

    text = f"🧠 Gemma의 수면 패턴 분석\n\n{analysis}"
    for chunk_start in range(0, len(text), 3500):
        await update.message.reply_text(text[chunk_start : chunk_start + 3500])


def _settings_keyboard(auto_volume: bool) -> InlineKeyboardMarkup:
    label = (
        "수면 감지 자동 볼륨 조절: 🟢 ON"
        if auto_volume
        else "수면 감지 자동 볼륨 조절: ⚪ OFF"
    )
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(label, callback_data="settings:toggle:auto_volume")]]
    )


def _settings_text(auto_volume: bool) -> str:
    state = "ON ✅" if auto_volume else "OFF"
    return (
        "⚙️ 봇 설정\n\n"
        f"- 수면 감지 자동 볼륨 조절: {state}\n\n"
        "아래 버튼을 눌러 ON/OFF를 전환할 수 있어요."
    )


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/settings — 사용자별 봇 기능 ON/OFF 토글 화면."""
    user_id = update.effective_user.id
    auto_volume = await get_auto_volume(user_id)
    await update.message.reply_text(
        _settings_text(auto_volume),
        reply_markup=_settings_keyboard(auto_volume),
    )


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """설정 토글 버튼 콜백."""
    query = update.callback_query
    await query.answer()

    parts = (query.data or "").split(":")
    if len(parts) != 3 or parts[0] != "settings" or parts[1] != "toggle":
        return

    user_id = update.effective_user.id

    if parts[2] == "auto_volume":
        current = await get_auto_volume(user_id)
        new_value = not current
        try:
            await set_auto_volume(user_id, new_value)
        except Exception:
            logger.exception("설정 저장 중 오류 발생")
            await query.message.reply_text("⚠️ 설정 저장 중 문제가 발생했습니다.")
            return
        await query.edit_message_text(
            _settings_text(new_value),
            reply_markup=_settings_keyboard(new_value),
        )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """대화 취소"""
    await update.message.reply_text(
        "상태 입력을 취소했습니다. 안녕히 주무세요! ✨", reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

def main() -> None:
    """봇 실행"""
    init_db()

    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            FATIGUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_fatigue)],
            STRESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_stress)],
            CAFFEINE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_caffeine)],
            MOOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_mood)],
            WAKE_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_wake_time)],
            TARGET_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_target_time)],
            SLEEP_LATENCY: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_sleep_latency)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CallbackQueryHandler(feedback_callback, pattern=r"^feedback:"))
    application.add_handler(CallbackQueryHandler(settings_callback, pattern=r"^settings:"))

    print("Sleep Agent Bot이 실행 중입니다... (종료하려면 Ctrl+C)")
    application.run_polling()

if __name__ == "__main__":
    main()
