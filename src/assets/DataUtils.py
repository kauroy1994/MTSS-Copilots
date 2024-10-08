import re

class AssetLoader:

	@staticmethod
	def get_queries():

		school_admin_queries = ["Briefly describe MTSS and the process of integrating behavioral health support into an existing academic framework.",
								"Discuss the importance of integrating behavioral health into a tiered academic support framework.",
								"Discuss the barriers to implement an integrated MTSS framework.",
								"What are the key concepts in implementing an effective integrated MTSS?",
								"We are just starting with MTSS. What are the core features of the framework. Where should we begin?",
								"I need family voice that represents our diverse school community.  I have lots of input from a vocal but small minority of parents expressing concerns about integrating positive health and wellness (SEL) in our Tier 1 curriculum. I believe most families are in favor of this, but they are not as vocal. What can I do to get more diverse voices especially around positive health and wellness (SEL)?",
								"We have seen an uptick in students coming to the school counselors expressing issue related to anxiety and depression. We currently refer to a community-based clinic for support but the students are not making it to their appointments for various reasons.  We would like to hire a clinician to work in the building. However, there is a group of parents in our district who are resistant to our school providing mental health services.  Can you tell me why it is important to deliver mental health services in schools? How can we increase buy in from the parents in our district?",
								"Why do schools struggle to integrate school behavioral health services into their MTSS",
								"Can you share with me, in a simple way, what RTI (Response to Intervention) is? How is it different than MTSS?",
								"What are the key concepts in implementing an effective integrated MTSS?"]

		clinical_staff_queries = ["What is MTSS, PBIS, ISF, RTI?  All these acronyms are floating around.  I need succinct answers. My next meeting is in 15 minutes.",
		"I am new to my district and have been invited to join an MTSS team as clinical staff. I am not clear on what they mean when they say I am an MTSS Team member. What is expected of me on this team?",
		"List the responsibilities of an LPC, LMSW, LMFT within a school building integrated MTSS framework.",
		"I work in a large school district.  It seems like every student who has an issue, behavior, emotion, and/or mental health is immediately referred to me.  I cant keep up.  What can I do to inform a more preventative approach in my school?",
		"My school does not use a universal screening system. What screeners can I recommend for use that capture internalizing and externalizing behavior? Can you give me a list of all available screening tools and a description of each."]

		queries = {"School_Administrators": school_admin_queries,
		"Clinical_Staff": clinical_staff_queries}

		return queries

	@staticmethod
	def get_templates():


		system_templates = {
			"School_Administrators": """You assist School Administrators whose job role is to ensures there is an MTSS team to design the school-wide implementation process, progress monitoring protocols, and data collection procedures.""",
			"Clinical_Staff": """You assist Clinical Staff whose job role involves the following:
							- attend all MTSS meetings
							-cprepare and report on qualitative and quantitative progress data on students and families receiving intensive support
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
			"School Psychologist":"""You assist School Administrators whose job role involves the following:
								- Contribute expertise in data interpretation and analysis, progress monitoring, and effective problem-solving.  
								- Administer diagnostic screening assessments assist in observing students in the
								- Instructional environment assist in designing interventions matched to student needs, based on data  
								- Assist with the identiﬁcation of appropriate interventions and progress monitoring  
								- Consult with the school-based leadership team and school staff regardi-solving ng MTSS needs 
								- Provide consultation and support to the school throughout the problemphases.""",
			"Teachers":"""You assist Teachers whose job role involves the following:
						- Provide high-quality standard-based instruction and interventions with ﬁdelity  
						- Implement selected schoolwide evidenced-based practices with ﬁdelity  
						- Collect data on the effectiveness of interventions
						- Collaborate in problem-solving efforts to determine interventions and supports  
						- Implement strategies, support, and plans for small group and individual students"""
				
			}

		return system_templates

	@staticmethod
	def read_data():

		f = open('/teamspace/studios/this_studio/MTSS-Copilots/src/assets/Final_txt_document_course.txt')
		f_lines = f.read().splitlines()
		f_str = ''.join([re.sub(r'[^A-Za-z0-9 ]+', '' ,line) for line in f_lines if re.sub(r'[^A-Za-z0-9 ]+', '' ,line)])
		f.close()
		return f_str