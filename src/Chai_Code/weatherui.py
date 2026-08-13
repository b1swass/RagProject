from dotenv import load_dotenv
load_dotenv()

import requests
import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Weather AI",
    page_icon="🌤️",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .weather-card {
        padding: 35px;
        border-radius: 25px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }

    .weather-city {
        font-size: 24px;
        font-weight: 600;
    }

    .weather-icon {
        font-size: 80px;
        margin: 10px 0;
    }

    .weather-temp {
        font-size: 65px;
        font-weight: 700;
    }

    .weather-condition {
        font-size: 20px;
        opacity: 0.9;
    }

    .weather-info {
        display: flex;
        gap: 50px;
        margin-top: 25px;
        font-size: 17px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather of a city.
    """

    print(f"Tool Called: get_weather({city})")

    url = f"https://wttr.in/{city}?format=%C|%t|%h|%w"

    try:

        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code == 200:
            return response.text.strip()

        return "Unable to get weather information."

    except requests.RequestException as e:

        return f"Weather API error: {e}"


# ============================================================
# LLM
# ============================================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)


# Give the model access to the tool
llm_with_tools = llm.bind_tools(
    [get_weather]
)


# ============================================================
# WEATHER STYLE
# ============================================================

def get_weather_style(
    condition: str,
    temperature: str
):

    condition = condition.lower()

    try:
        temp = float(
            temperature
            .replace("°C", "")
            .replace("°F", "")
            .strip()
        )

    except ValueError:
        temp = 20


    # Thunderstorm
    if "thunder" in condition:

        return (
            "⛈️",
            "Thunderstorm",
            "linear-gradient(135deg, #232526, #414345)"
        )


    # Rain
    if "rain" in condition or "drizzle" in condition:

        return (
            "🌧️",
            "Rainy",
            "linear-gradient(135deg, #4b6cb7, #182848)"
        )


    # Snow
    if "snow" in condition:

        return (
            "❄️",
            "Snowy",
            "linear-gradient(135deg, #83a4d4, #b6fbff)"
        )


    # Fog / Mist
    if "fog" in condition or "mist" in condition:

        return (
            "🌫️",
            "Foggy",
            "linear-gradient(135deg, #757F9A, #D7DDE8)"
        )


    # Cloudy
    if "cloud" in condition or "overcast" in condition:

        return (
            "☁️",
            "Cloudy",
            "linear-gradient(135deg, #536976, #292E49)"
        )


    # Very hot
    if temp >= 30:

        return (
            "☀️",
            "Hot & Sunny",
            "linear-gradient(135deg, #f7971e, #ffd200)"
        )


    # Very cold
    if temp <= 10:

        return (
            "🥶",
            "Very Cold",
            "linear-gradient(135deg, #2c3e50, #4ca1af)"
        )


    # Normal
    return (
        "🌤️",
        "Pleasant Weather",
        "linear-gradient(135deg, #56ccf2, #2f80ed)"
    )


# ============================================================
# WEATHER CARD
# ============================================================

def show_weather_card(
    city: str,
    weather_data: str
):

    try:

        condition, temperature, humidity, wind = (
            weather_data.split("|")
        )

    except ValueError:

        st.error(
            "Unable to parse weather information."
        )

        return


    icon, title, background = get_weather_style(
        condition,
        temperature
    )


    st.markdown(
        f"""
        <div
            class="weather-card"
            style="background: {background};"
        >

            <div class="weather-city">
                {city}
            </div>

            <div class="weather-icon">
                {icon}
            </div>

            <div class="weather-condition">
                {title}
            </div>

            <div class="weather-temp">
                {temperature}
            </div>

            <div class="weather-condition">
                {condition}
            </div>

            <div class="weather-info">

                <div>
                    💧 Humidity<br>
                    <b>{humidity}</b>
                </div>

                <div>
                    💨 Wind<br>
                    <b>{wind}</b>
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.title("🌤️ Weather AI")

st.caption(
    "Ask me about the current weather anywhere in the world."
)


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask something..."
)


# ============================================================
# CONTINUOUS INPUT
# ============================================================

if query:

    # Exit
    if query.strip() == "0":

        st.success("Goodbye! 👋")
        st.stop()


    # ========================================================
    # USER
    # ========================================================

    with st.chat_message("user"):

        st.write(query)


    # ========================================================
    # AI
    # ========================================================

    with st.chat_message("assistant"):


        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        with st.expander(
            "🟢 START",
            expanded=False
        ):

            st.write(
                "Understanding the user query..."
            )


        # ----------------------------------------------------
        # PLAN
        # ----------------------------------------------------

        with st.expander(
            "🧠 PLAN",
            expanded=False
        ):

            st.write(
                "Determining whether a tool is required..."
            )


        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        response = llm_with_tools.invoke(
            [
                HumanMessage(
                    content=f"""
You are a helpful AI assistant.

You have access to tools.

Rules:

- Answer the user clearly.
- Use a tool when necessary.
- Never invent tool results.
- Never output HTML.
- Never output CSS.
- Return normal plain text.

User query:

{query}
"""
                )
            ]
        )


        # ====================================================
        # TOOL CALL
        # ====================================================

        if response.tool_calls:

            tool_messages = []


            with st.expander(
                "⚡ ACTION",
                expanded=True
            ):

                for tool_call in response.tool_calls:

                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    st.code(
                        f"{tool_name}({tool_args})"
                    )


                    # ----------------------------------------
                    # GET WEATHER
                    # ----------------------------------------

                    if tool_name == "get_weather":

                        city = tool_args["city"]


                        weather_result = (
                            get_weather.invoke(
                                tool_args
                            )
                        )


                        # ------------------------------------
                        # OBSERVE
                        # ------------------------------------

                        with st.expander(
                            "👀 OBSERVE",
                            expanded=False
                        ):

                            st.code(
                                weather_result
                            )


                        # ------------------------------------
                        # WEATHER CARD
                        # ------------------------------------

                        show_weather_card(
                            city,
                            weather_result
                        )


                        # ------------------------------------
                        # TOOL MESSAGE
                        # ------------------------------------

                        tool_messages.append(
                            ToolMessage(
                                content=weather_result,
                                tool_call_id=tool_call["id"]
                            )
                        )


            # =================================================
            # FINAL LLM RESPONSE
            # =================================================

            final_response = (
                llm_with_tools.invoke(
                    [
                        HumanMessage(
                            content=query
                        ),
                        response,
                        *tool_messages
                    ]
                )
            )


            # =================================================
            # OUTPUT
            # =================================================

            st.markdown("### 🤖 OUTPUT")

            st.write(
                final_response.content
            )


        # ====================================================
        # NO TOOL
        # ====================================================

        else:

            st.markdown("### 🤖 OUTPUT")

            st.write(
                response.content
            )   