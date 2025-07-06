from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent   
import os  
import glob
from google.adk.agents.readonly_context import ReadonlyContext

  
# 파일 읽기 및 요약 도구  
def read_file_content(file_path: str) -> str:  
    with open(file_path, 'r', encoding='utf-8') as f:  
        content = f.read()  
    return content
  
# 50개 파일을 10개씩 5그룹으로 분할  
all_files = glob.glob("data/*.md") 

# 각 그룹을 처리할 에이전트 생성  
summary_agents = []  
for i, file in enumerate(all_files):  
    output_key = f"summary_{i+1}"
    content = read_file_content(file)
    agent = LlmAgent(  
        name=f"file_summarizer_{i+1}",  
        model="gemini-2.5-flash-preview-05-20",  
        description=f"{i+1}번째 대본 {file} 요약하는 에이전트",  
        output_key=output_key,
        instruction=f"""  

        ## 목표:
        할당된 대본을 읽고 개발자의 성장에 관하여 도움이 되는 내용만 추려주는 에이전트입니다.
        "최고의 개발자가 되는 법"이라는 주제에 대해 강력한 상관관계가 있는 문장만 추리고, 그중에서도 상, 중, 하로 추가적으로 표기해 주세요. "최고의 개발자가 되는 법"이라는 주제와 관계있는 부분이 없다면 관계없음 이라고 적어주세요.

        ## 응답형식:
        \`\`\`json
        {{
          "fileName": {file},
          "key": [
          {{
            content: 개발자성장에 도움되는 문장과 타임스탬프,
            relation: 중
          }},
          {{
            content: 개발자성장에 도움되는 문장과 타임스탬프,
            relation: 상
          }},
          ...
        ]
        }}
        \`\`\`

        
        ## 분석할 대본:
        {content}
        """,  
    )  
    summary_agents.append(agent)  
  

def build_final_instruction(readonly_context: ReadonlyContext) -> str:  
    # 동적으로 생성된 키들로 세션 상태에서 데이터 가져오기  
    summaries = []  
    for i in range(len(all_files)):  
        key = f"summary_{i+1}"  
        if key in readonly_context.state:  
            summaries.append(readonly_context.state[key])  
      
    return f""" 
    ## 목표
    하위 에이전트들이 "최고의 개발자가 되는법" 과 관련된 문장들을 추려줄 예정입니다.
    이중에 공통적으로 자주 언급되는 내용을 찾아, "최고의 개발자가 되는법"에 대한 중요한 내용을 도출해주세요.
    
    ## 방법
    1. 수많은 문장들중 공통적으로 가장 자주 언급되는 공통된 내용에는 어떤것이 있는지 찾아주세요
    2. 그 내용을 내포하는 문장과 출처들을 배열로 담아주세요
    3. 몇번 반복되었는지 적어주세요

    ## 출력형식:
    \`\`\`json
    {{
      conclusion: 개발자성장에 도움되는 문장과 타임스탬프,
      contents: [해당 내용을 내포하는 문장들을 배열로 담아주세요]
      count: 몇번 반복되었는지 적어주세요.
    }}
    \`\`\`

    ## 관련된 문장들: 
    {chr(10).join(summaries)}  
    """  



# 병렬 처리 에이전트  
parallel_summarizer = ParallelAgent(  
    name="parallel_file_processor",  
    sub_agents=summary_agents  
)  
  
# 최종 분석 에이전트  
final_analyzer = LlmAgent(  
    name="final_analyzer",  
    model="gemini-2.5-pro",  
    description="5개의 리포트를 취합하여 최종 분석",  
    instruction=build_final_instruction
)  
  
# 전체 워크플로우를 조율하는 루트 에이전트  
root_agent = SequentialAgent(  
    name="document_analysis_agent",  
    description="병렬처리 요약 후 최종 분석",  
    sub_agents=[parallel_summarizer, final_analyzer],  
)