from google.adk.agents import LlmAgent, ParallelAgent  
from google.adk.tools import FunctionTool  
import os  
import glob  
  
# 파일 읽기 및 요약 도구  
def read_and_summarize_files(file_path: str) -> str:  
    """파일들을 읽고 요약합니다."""  
    summaries = []  
    with open(file_path, 'r', encoding='utf-8') as f:  
        content = f.read()  
        # 여기서 실제 요약 로직 구현  
        summary = f"파일 {file_path}중 유의미한 내용: {content[:200]}..."  
        summaries.append(summary)  
    return "\n".join(summaries)  
  
# 50개 파일을 10개씩 5그룹으로 분할  
all_files = glob.glob("data/*.md") 

# 각 그룹을 처리할 에이전트 생성  
summary_agents = []  
for i, file in enumerate(all_files):  
    agent = LlmAgent(  
        name=f"file_summarizer_{i+1}",  
        model="gemini-2.0-flash",  
        description=f"파일 {i+1}을 요약하는 에이전트",  
        instruction=f"""  
        할당된 1개의 대본을 읽고 개발자의 성장에 관하여 도움이 되는 내용만 추려주는 에이전트입니다.
        개발자의 성장에 관하여 도움이 되는 문장만 추리고, 해당 문장들이 개발자의 성장이라는 주제와 얼마나 연관성이 있는지 상, 중, 하로 추가적으로 표기해 주세요.
        주어진 대본이 개발자의 성장과 무관한 내용으로 구성되어있다면 무관한 내용이라고 표기해 주세요.
        """,  
        tools=[lambda path=file: read_and_summarize_files(path)]  
    )  
    summary_agents.append(agent)  
  
# 병렬 처리 에이전트  
parallel_summarizer = ParallelAgent(  
    name="parallel_file_processor",  
    sub_agents=summary_agents  
)  
  
# 최종 분석 에이전트  
final_analyzer = LlmAgent(  
    name="final_analyzer",  
    model="gemini-2.0-flash",  
    description="5개의 리포트를 취합하여 최종 분석",  
    instruction="""  
    5개의 하위 에이전트가 생성한 리포트를 받아서:  
    1. 공통 패턴과 트렌드 식별  
    2. 주요 발견사항 요약  
    3. 전체적인 결론 도출  
    4. 실행 가능한 권장사항 제시  
    """  
)  
  
# 전체 워크플로우를 조율하는 루트 에이전트  
root_agent = LlmAgent(  
    name="document_analysis_coordinator",  
    model="gemini-2.0-flash",  
    description="문서 분석 프로세스를 조율",  
    sub_agents=[parallel_summarizer, final_analyzer],  
    instruction="""
    사용자가 분석시작 이라고 말하면,
    먼저 병렬 처리 에이전트를 통해 파일들에서 유의미한 내용을 뽑아내고,  
    그 결과를 최종 분석 에이전트에게 전달하여 종합 분석을 수행하세요.  
    """  
)