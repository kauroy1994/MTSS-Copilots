import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Define system templates
system_templates = {
    "School_Administrators": """You assist School Administrators whose job role is to ensure there is an MTSS team to design the school-wide implementation process, progress monitoring protocols, and data collection procedures.""",
    "Clinical_Staff": """You assist Clinical Staff whose job role involves the following:
                        - attend all MTSS meetings
                        - prepare and report on qualitative and quantitative progress data on students and families receiving intensive support
                        - facilitate problem solving, offering key insights on the impacts and potential root causes of internalizing and externalizing behaviors
                        - identify and collaborate on research-based intervention strategies implemented by school staff
                        - support problem-solving and mediation for educators 
                        - lead and plan professional development related to individual and systems level mental health and wellness strategies 
                        - facilitate small groups related to prevention, intervention, and postvention support
                        - continuously interact with and engage families at both the community and individual level
                        """,
    "School_Counselor": """You assist School Counselors whose job role involves the following:
                            - Act as a primary resource for administrators, teachers, and parents regarding mental health awareness
                            - Implement school counseling programs addressing the needs of all students. 
                            - Deliver instruction, appraisal, and advisement to students in all tiers and collaborate with other specialized instructional and intervention personnel, educators, and families to ensure appropriate academic and behavioral supports for students within the school’s MTSS framework. 
                            - Providing all students with standards-based school counseling instruction to address universal academic, career, and social/emotional development and
                            - Analyzing academic, career, and social/emotional development data to identify students who need support
                            - Identifying and collaborating on research-based intervention strategies implemented by school staff
                            - Evaluating academic and behavioral progress after interventions
                            - Revising interventions as appropriate
                            - Referring to school and community services as appropriate
                            - Collaborating with administrators, teachers, other school professionals, community agencies, and families in MTSS design and implementation 
                            - Advocating for equitable education for all students and working to remove systemic barriers""",
    "School_Psychologist": """You assist School Psychologists whose job role involves the following:
                            - Contribute expertise in data interpretation and analysis, progress monitoring, and effective problem-solving.  
                            - Administer diagnostic screening assessments and assist in observing students in the
                            - Instructional environment and assist in designing interventions matched to student needs, based on data  
                            - Assist with the identiﬁcation of appropriate interventions and progress monitoring  
                            - Consult with the school-based leadership team and school staff regarding MTSS needs 
                            - Provide consultation and support to the school throughout the problem-solving phases.""",
    "Teachers": """You assist Teachers whose job role involves the following:
                    - Provide high-quality standard-based instruction and interventions with ﬁdelity  
                    - Implement selected schoolwide evidenced-based practices with ﬁdelity  
                    - Collect data on the effectiveness of interventions
                    - Collaborate in problem-solving efforts to determine interventions and supports  
                    - Implement strategies, support, and plans for small group and individual students"""
}

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_keywords(case_text, n=5):
    """
    Extracts keywords from the case text using TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=n, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([case_text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

def format_role_duties_html(relevant_roles_duties):
    """
    Formats the duties of relevant roles as an HTML string for better visualization.
    """
    html_content = "<div style='font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6;'>"
    for role, duties in relevant_roles_duties.items():
        html_content += f"<h3 style='color: #2e6da4;'>{role}</h3>"
        html_content += f"<p>{duties}</p><hr>"
    html_content += "</div>"
    return html_content

def visualize_roles(case_text):
    # Embed role descriptions
    role_embeddings = []
    roles = []
    
    for role, description in system_templates.items():
        roles.append(role)
        role_embeddings.append(model.encode(description))
    
    # Convert role_embeddings to a NumPy array
    role_embeddings = np.array(role_embeddings)
    
    # Embed the case text and extract key phrases
    case_embedding = model.encode(case_text)
    keywords = extract_keywords(case_text)
    keyword_embeddings = [model.encode(keyword) for keyword in keywords]
    
    # Combine all embeddings for T-SNE
    all_embeddings = np.vstack((role_embeddings, keyword_embeddings))
    all_labels = roles + keywords
    
    # Calculate similarities between case embedding and role embeddings
    similarities = np.dot(role_embeddings, case_embedding) / (np.linalg.norm(role_embeddings, axis=1) * np.linalg.norm(case_embedding))
    
    # Perform T-SNE on all embeddings
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
    tsne_results = tsne.fit_transform(all_embeddings)
    
    # Plot the T-SNE results
    plt.figure(figsize=(12, 10))
    for i, (x, y) in enumerate(tsne_results):
        if i < len(roles):
            color = 'red' if similarities[i] > 0.3 else 'blue'
            plt.scatter(x, y, color=color, s=100)
            plt.text(x + 0.02, y + 0.02, roles[i], fontsize=12, ha='right')
        else:
            plt.scatter(x, y, color='green', s=50)
            plt.text(x + 0.02, y + 0.02, keywords[i - len(roles)], fontsize=12, ha='right')
    
    # Draw connections between relevant keywords and roles
    for i, (x_role, y_role) in enumerate(tsne_results[:len(roles)]):
        for j, (x_kw, y_kw) in enumerate(tsne_results[len(roles):]):
            if similarities[i] > 0.3:  # Only connect relevant roles
                plt.plot([x_role, x_kw], [y_role, y_kw], color='gray', linestyle='--', linewidth=0.5)
    
    plt.title("T-SNE Visualization of Roles, Duties, and Case Keywords")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    
    # Save and display the plot
    plot_path = "tsne_roles_case.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Retrieve the most relevant roles and their descriptions
    relevant_roles = [roles[i] for i in range(len(roles)) if similarities[i] > 0.3]
    relevant_descriptions = {role: system_templates[role] for role in relevant_roles}
    
    # Format the duties for HTML display
    relevant_roles_html = format_role_duties_html(relevant_descriptions)
    
    return plot_path, case_text, relevant_roles_html

def process_case(case_text):
    plot_path, original_case, relevant_roles_html = visualize_roles(case_text)
    return plot_path, original_case, relevant_roles_html

# Gradio Interface
def main_interface():
    gr.Interface(
        fn=process_case,
        inputs=gr.Textbox(lines=10, placeholder="Enter a case scenario...", label="Case Scenario"),
        outputs=[
            gr.Image(type="filepath", label="T-SNE Visualization of Roles"),
            gr.HTML(label="Original Case Scenario"),
            gr.HTML(label="Duties of Relevant Roles")
        ],
        title="T-SNE Visualization of Relevant Roles, Duties, and Case Elements",
        description=(
            "Enter a case scenario, and the application will visualize relevant roles, "
            "duties, and extracted keywords from the case using T-SNE. Roles closely aligned "
            "with the case will be highlighted in red, and their detailed duties will be displayed below."
        ),
        theme="default",
        layout="vertical"
    ).launch()

# Run the interface
if __name__ == "__main__":
    main_interface()
