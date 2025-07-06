from google.adk.agents import LlmAgent, ParallelAgent  
from google.adk.tools import FunctionTool  
import os  
import glob  
  
# 파일 읽기 및 요약 도구  
def read_and_summarize_files(file_paths: list[str]) -> str:  
    """파일들을 읽고 요약합니다."""  
    summaries = []  
    for file_path in file_paths:  
        with open(file_path, 'r', encoding='utf-8') as f:  
            content = f.read()  
            # 여기서 실제 요약 로직 구현  
            summary = f"파일 {file_path}의 요약: {content[:200]}..."  
            summaries.append(summary)  
    return "\n".join(summaries)  
  
# 50개 파일을 10개씩 5그룹으로 분할  
all_files = glob.glob("data/*.md")[:50]  
file_groups = [all_files[i:i+10] for i in range(0, 50, 10)]  
  
# 각 그룹을 처리할 에이전트 생성  
summary_agents = []  
for i, file_group in enumerate(file_groups):  
    agent = LlmAgent(  
        name=f"file_summarizer_{i+1}",  
        model="gemini-2.0-flash",  
        description=f"파일 그룹 {i+1}을 요약하는 에이전트",  
        instruction=f"""  
        할당된 {len(file_group)}개의 파일을 읽고 각각 요약한 후,  
        전체적인 패턴과 주요 내용을 분석하여 종합 리포트를 작성하세요.  
        """,  
        tools=[lambda paths=file_group: read_and_summarize_files(paths)]  
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
    먼저 병렬 처리 에이전트를 통해 50개 파일을 5개 그룹으로 나누어 처리하고,  
    그 결과를 최종 분석 에이전트에게 전달하여 종합 분석을 수행하세요.  
    """  
)