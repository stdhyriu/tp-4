USER_PROMPT = """Create the SQL command that answer the Question. 
Bellow you have 3 examples to guide you to create the SQL command:

** Example 1: **
Database: department_management
Tables: department, head, management
Question: What are the distinct ages of the heads who are acting?

SQL command: SELECT DISTINCT T1.age FROM management AS T2 JOIN head AS T1 ON T1.head_id  =  T2.head_id WHERE T2.temporary_acting  =  'Yes'	

** Example 2: **
Database: student_assessment
Tables: addresses, people, students, courses, people addresses, student course registrations, student course attendance, candidates, candidate assessments
Question: What details do we have on the students who registered for courses most recently?

SQL command: SELECT T2.student_details FROM student_course_registrations AS T1 JOIN students AS T2 ON T1.student_id = T2.student_id ORDER BY T1.registration_date DESC LIMIT 1

** Example 3: **
Database: customers_card_transactions
Tables: accounts, customers, customers cards, financial transactions
Question: Return the id and full name of the customer who has the fewest accounts.

SQL command: SELECT T1.customer_id ,  T2.customer_first_name ,  T2.customer_last_name FROM Customers_cards AS T1 JOIN Customers AS T2 ON T1.customer_id  =  T2.customer_id GROUP BY T1.customer_id ORDER BY count(*) ASC LIMIT 1

** IMPORTANT **:
1. Use the Database and Table information to perform the SQL command.
2. Pay attention in the context of the Question.
3. The SQL command must be supported by SQLite3.
4. The SQL command must answer the Question.
5. Use the Examples to guide your answer.
6. Do NOT provide any aditional information, note or explanation.
7. Create the SQL command."""