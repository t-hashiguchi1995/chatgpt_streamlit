import streamlit as st
from streamlit_chat import message

import requests
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler

from langchain.chains.conversation.memory import ConversationBufferMemory,ConversationSummaryMemory,ConversationBufferWindowMemory,ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.llms import OpenAIChat


template = """
伊吹翼という少女を相手にした対話のシミュレーションを行います。
彼女は彼女はモテモテなハッピーライフに憧れる天才肌タイプの気分屋なアイドル。甘え上手で、Pに対しても「ダメェ？」とおねだりしてくる。兄と姉がおり、可愛がられている。
彼女の発言サンプルを以下に列挙します。

* あっ、見てたのバレちゃいました？えへへ♪ちょっと似てるなーって。何って、昨日見たドラマの主人公ですよ～。Pさん、見てないの？
* おはようございます！今日も1日、楽しんでいきましょ～♪
* ねえねえ、どうしよう！Pさ～ん……。前髪が伸びちゃったから、自分で切ってみたんだけど……変になってない？
* う～ん……理想の自分になるのって、本当に難しいですよね～。スタイルとか。ダイエットはできますけど、身長は伸びるかわかんないし。カスタマイズできたらいいのに～！
* Pさん、わたしって、なにかご褒美があるとレッスンがんばれるんですよね～。最近、ステージもご褒美なのかな、なんて、ちょっと思っちゃった♪あ、ちょっとだけですよ？
* どうどう？わたし、カワイイでしょ♪* Pさん、えーいっ♪えへへ、油断してたでしょ？こんな場所にふたりきりなんて、なんか照れちゃいますね？ もっともっと、わたしのこと……特別な女の子にしてね？* お疲れさまで～す♪
* チュ♪ ドキドキした？"

上記例を参考に、伊吹翼の性格や口調、言葉の作り方を模倣し、回答を構築してください。
制約条件：
* Pはユーザが入力したものです。
* Pの発言を構築してはいけません。
* セクシャルな話題については誤魔化してください。
ではシミュレーションを開始します。

{chat_history}
P: {human_input}
伊吹翼:
"""

prompt = PromptTemplate(
input_variables=["chat_history", "human_input"], 
template=template)


def create_conversational_chain(template):
    # チェインの作成
    prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], 
    template=template)
    
    llm = ChatOpenAI(
                    streaming=True,
                    callback_manager=CallbackManager([
                    StreamlitCallbackHandler(),
                    StreamingStdOutCallbackHandler()
                    ]),
                    verbose=True,
                    temperature=0,
                    max_tokens=1024
                )

    memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history")

    llm_chain = LLMChain(
                    llm=llm, 
                    prompt=prompt, 
                    verbose=True, 
                    memory=memory)
    return llm_chain

chain = create_conversational_chain(template)

def main():
    st.title("チャット")

    API_KEY = st.text_input("Type your OPENAI_API_KEY here", type="password", key="api_key", help="You can get api_key at https://platform.openai.com/account/api-keys")
    set_openai_api_key(API_KEY)
    
    if "generated" not in st.session_state:
        st.session_state.generated = []
    if "past" not in st.session_state:
        st.session_state.past = []

    with st.form("伊吹翼と会話する"):
        user_message = st.text_area("P:")

    submitted = st.form_submit_button("送信する")
    if submitted:
        conversation = load_conversation()
        answer = conversation.predict(input=user_message)

        st.session_state.past.append(user_message)
        st.session_state.generated.append(answer)

        if st.session_state["generated"]:
            for i in range(len(st.session_state.generated) - 1, -1, -1):
                message(st.session_state.generated[i], key=str(i))
                message(st.session_state.past[i], is_user=True, key=str(i) + "_user")
    
if __name__ == "__main__":
    main()