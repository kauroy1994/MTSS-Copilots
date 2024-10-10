import os
import pandas as pd
import tiktoken
import asyncio
from rich import print
import json
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# 1. Setup LLM
api_key = os.environ["GRAPHRAG_API_KEY"]
llm_model = "phi3.5"
embedding_model = "text-embedding-3-small"

# role = "nurse"

# system_prompt = """You have been asked a question by Clinical_Staff whose job role is to ensure mental health across all tiers of the MTSS. Clinical_Staff is a key member of both the MTSS team and the school culture. Their visibility and voice on this team and throughout the building communicate the importance of mental health across all tiers of the MTSS. They function as not only a referral source for students and families with more intensive needs, but also as a consultant for wellness efforts across the tiers. School mental health clinicians:
# - attend all MTSS meetings
# - prepare and report on qualitative and quantitative progress data on students and families receiving intensive support
# - facilitate problem solving, offering key insights on the impacts and potential root causes of internalizing and externalizing behaviors.
# """
# print(type(system_prompt))

# Define system templates
system_templates = {
    "admin": """You have been asked a question by School Administrators who job role is to ensures there is an MTSS team to design the school-wide implementation process, progress monitoring protocols, and data collection procedures.""",
    "clinician": """You have been asked a question by Clinical_Staff whose job role is to ensure mental health across all tiers of the MTSS. Clinical_Staff is a key member of both the MTSS team and the school culture. Their visibility and voice on this team and throughout the building communicate the importance of mental health across all tiers of the MTSS. They function as not only a referral source for students and families with more intensive needs, but also as a consultant for wellness efforts across the tiers. School mental health clinicians:
    - attend all MTSS meetings
    - prepare and report on qualitative and quantitative progress data on students and families receiving intensive support
    - facilitate problem solving, offering key insights on the impacts and potential root causes of internalizing and externalizing behaviors.""",
    "counselor": """You have been asked a question by School Counselors whose job role is to As a primary resource for administrators, teachers, and parents regarding mental health awareness, school counselors implement school counseling programs addressing the needs of all students. School counselors deliver instruction, appraisal, and advisement to students in all tiers and collaborate with other specialized instructional and intervention personnel, educators, and families to ensure appropriate academic and behavioral supports for students within the school’s MTSS framework. According to the American School Counselor Association, school counselors align with the school’s MTSS by:
    - providing all students with standards-based school counseling instruction to address universal academic, career, and social/emotional development and analyzing academic, career, and social/emotional development data to identify students who need support
    - identifying and collaborating on research-based intervention strategies implemented by school staff
    - evaluating academic and behavioral progress after interventions
    - revising interventions as appropriate
    - referring to school and community services as appropriate
    - collaborating with administrators, teachers, other school professionals, community agencies, and families in MTSS design and implementation 
    - advocating for equitable education for all students and working to remove systemic barriers""",
    'psychologist': """You have been asked a question by School Administrators whose job role is to School psychologists play an integral role in promoting and supporting competency development within the core components of MTSS, including data-informed decision making, evidence-based interventions, implementation ﬁdelity, and consultation and collaboration (National Association of School Psychologists, 2020). In MTSS, data-informed decision making includes universal screening of all students, implementation of evidenced-based interventions at multiple tiers, and ongoing progress monitoring to inform the decisions at each tier. A problem-solving process supports ongoing evaluation of the data in order to make timely and ongoing informed decisions (Gresham, 2007). School psychologists 
    - contribute expertise in data interpretation and analysis, progress monitoring, and effective problem-solving.  
    - administer diagnostic screening assessments assist in observing students in the instructional environment assist in designing interventions matched to student needs, based on data  
    - assist with the identiﬁcation of appropriate interventions and progress monitoring  
    - consult with the school-based leadership team and school staff regarding MTSS needs 
    - provide consultation and support to the school throughout the problem-solving phases.""",
    'teacher': """You have been asked a question by School Administrators whose job role is to
    - provide high-quality standard-based instruction and interventions with ﬁdelity  
    - implement selected schoolwide evidenced-based practices with ﬁdelity  
    - collect data on the effectiveness of Tier 1, Tier 2, and Tier 3 interventions (progress monitoring)  
    - collaborate in problem-solving efforts to determine interventions and supports  
    - implement strategies, support, and plans for small group and individual students
    - ensure that appropriate data are"""
}


llm = ChatOpenAI(api_key="ollama", model=llm_model, api_type=OpenaiApiType.OpenAI, api_base="http://localhost:11434/v1", max_retries=20)
token_encoder = tiktoken.get_encoding("cl100k_base")
text_embedder = OpenAIEmbedding(api_key=api_key, api_type=OpenaiApiType.OpenAI, model=embedding_model, max_retries=20)

# 2. Load the Context
INPUT_DIR = "/Users/praison/graphrag/inputs/artifacts"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
TEXT_UNIT_TABLE = "create_final_text_units"
LANCEDB_URI = "ragtest/output/lancedb"
COMMUNITY_LEVEL = 2

entity_df = pd.read_parquet("ragtest/output/create_final_nodes.parquet")
entity_embedding_df = pd.read_parquet("ragtest/output/create_final_entities.parquet")
entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
description_embedding_store.connect(db_uri=LANCEDB_URI)
store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

relationship_df = pd.read_parquet("ragtest/output/create_final_relationships.parquet")
relationships = read_indexer_relationships(relationship_df)

report_df = pd.read_parquet("ragtest/output/create_final_community_reports.parquet")
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

text_unit_df = pd.read_parquet("ragtest/output/create_final_text_units.parquet")
text_units = read_indexer_text_units(text_unit_df)

context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,
    text_embedder=text_embedder,
    token_encoder=token_encoder,
)

# 4. Setup Local Search
local_context_params = { "text_unit_prop": 0.5, "community_prop": 0.1, "conversation_history_max_turns": 5, "conversation_history_user_turns_only": True, "top_k_mapped_entities": 10, "top_k_relationships": 10, "include_entity_rank": True, "include_relationship_weight": True, "include_community_rank": False, "return_candidate_context": False, "max_tokens": 12_000, }
llm_params = { "max_tokens": 2_000, "temperature": 0.0, }
search_engine = LocalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    llm_params=llm_params,
    context_builder_params=local_context_params,
    response_type="multiple paragraphs",
)

# 5. Run Local Search
async def run_search(query: str, system_prompt: str):
    full_query = system_prompt + "\n\n" + query
    result = await search_engine.asearch(full_query)
    return result

# 6. Question Generation
question_generator = LocalQuestionGen(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    llm_params=llm_params,
    context_builder_params=local_context_params,
)

async def generate_questions(history):
    questions = await question_generator.agenerate(question_history=history, context_data=None, question_count=5)
    return questions

# 7. Automate QA Process
def read_questions_from_file(file_path):
    with open(file_path, 'r') as file:
        questions = file.readlines()
    return [q.strip() for q in questions]

if __name__ == "__main__":
    for role, system_prompt in system_templates.items():
        questions_file = f"questions_{role}.txt"
        questions = read_questions_from_file(questions_file)
        
        responses = []
        supplementary_info = []

        for query in questions:
            result = asyncio.run(run_search(query, system_prompt))
            
            # Collect response data
            response_data = {
                "query": query,
                "response": result.response,
                "role": role  # Add the role field
            }
            responses.append(response_data)
            
            # Collect supplementary information
            supplementary_data = {
                "query": query,
                "entities": result.context_data["entities"].head().to_dict(),
                "relationships": result.context_data["relationships"].head().to_dict(),
                "reports": result.context_data["reports"].head().to_dict(),
                "sources": result.context_data["sources"].head().to_dict()
            }
            if "claims" in result.context_data:
                supplementary_data["claims"] = result.context_data["claims"].head().to_dict()
            
            supplementary_info.append(supplementary_data)
            
            # Print for debugging
            print(f"Role: {role}")
            print(f"Query: {query}")
            print(result.response)
            print(result.context_data["entities"].head())
            print(result.context_data["relationships"].head())
            print(result.context_data["reports"].head())
            print(result.context_data["sources"].head())
            if "claims" in result.context_data:
                print(result.context_data["claims"].head())
                print("\n")
        
        # Write responses to JSON file
        with open(f"{llm_model}_results/{role}_local_aligned_responses.json", "w") as f:
            json.dump(responses, f, indent=4)
        print(f'Responses written to mistral_results/{role}_aligned_local_responses.json')
        
        # Write supplementary information to JSON file
        with open(f"{llm_model}_results/{role}_local_aligned_sup_info.json", "w") as f:
            json.dump(supplementary_info, f, indent=4)
        print(f'Supplementary information written to mistral_results/{role}_aligned_local_sup_info.json')

# if __name__ == "__main__":
#     questions_file = "questions.txt"
#     questions = read_questions_from_file(questions_file)
    
#     for query in questions:
#         result = asyncio.run(run_search(query))
#         print(f"Query: {query}")
#         print(result.response)
#         print(result.context_data["entities"].head())
#         print(result.context_data["relationships"].head())
#         print(result.context_data["reports"].head())
#         print(result.context_data["sources"].head())
#         if "claims" in result.context_data:
#             print(result.context_data["claims"].head())
#         print("\n")

#     history = ["Future Work", "VLM"]
#     questions = asyncio.run(generate_questions(history))
#     print(questions.response)

# if __name__ == "__main__":
#     # role = "nurse"
#     for role, system_prompt in system_templates.items():
#         questions_file = f"questions_{role}.txt"
#         questions = read_questions_from_file(questions_file)
        
#         responses = []
#         supplementary_info = []

#         for query in questions:
#             result = asyncio.run(run_search(query))
            
#             # Collect response data
#             response_data = {
#                 "query": query,
#                 "response": result.response,
#                 "role": role
                
#             }
#             responses.append(response_data)
            
#             # Collect supplementary information
#             supplementary_data = {
#                 "query": query,
#                 "entities": result.context_data["entities"].head().to_dict(),
#                 "relationships": result.context_data["relationships"].head().to_dict(),
#                 "reports": result.context_data["reports"].head().to_dict(),
#                 "sources": result.context_data["sources"].head().to_dict()
#             }
#             if "claims" in result.context_data:
#                 supplementary_data["claims"] = result.context_data["claims"].head().to_dict()
            
#             supplementary_info.append(supplementary_data)
            
#             # Print for debugging
#             print(f"Role: {role}")
#             print(f"Query: {query}")
#             print(result.response)
#             print(result.context_data["entities"].head())
#             print(result.context_data["relationships"].head())
#             print(result.context_data["reports"].head())
#             print(result.context_data["sources"].head())
#             if "claims" in result.context_data:
#                 print(result.context_data["claims"].head())
#                 print("\n")
        
#         # Write responses to JSON file
#         with open(f"mistral_results/{role}_local_responses.json", "w") as f:
#             json.dump(responses, f, indent=4)
#         print('Responses written to responses.json')
        
#         # Write supplementary information to JSON file
#         with open(f"mistral_results/{role}_local_sup_info.json", "w") as f:
#             json.dump(supplementary_info, f, indent=4)
#         print('Supplementary information written to supplementary_info.json')

