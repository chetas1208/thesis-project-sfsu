import pdfplumber
import os
import sys
import re 

if sys.stdout.encoding != 'utf-8':
  import io

  sys.stdout = io.TextIOWrapper(
      sys.stdout.buffer, encoding='utf-8', errors='replace'
  )
#IMPORTANT: TALK ABOUT IT. HOW WOULD WE EXTRACT COURSE NAME AS SYLABBUSES VARY AND ARENT REALLY FOLLOWING A GENERIC FORMAT OF [COURSE NAME: CSC XYZ]
#FOR NOW WE CAN THINK ABOUT ALWAYS PICKING THE HIGHER CSC NUMBER AS PRE REQ'S LOOK LIKE TO ALWAYS BE A LOWER NUMBER 
#syllabi being some pdf stores in out inputs file
#NOTES: Should be able to work on many types of things 

#POTENTIAL ISSUES/ EDGE CASES: Alot of times most sylabusses contain "CSC XYZ = TITLE" however for some it follows a different format and just uses Title:class name seperate from the usual
def parse(syllabi):
    total=[]
    with pdfplumber.open(syllabi) as pdf:
        for page in pdf.pages:
            text=page.extract_text()

            total.append(text)

    return "\n".join(total)

def extract(parsedText):
    x = None
    y = None
    course_title_index=re.finditer("CSC",parsedText)

    for potentials_index in course_title_index:#return start index of CSC, need to gather say the next 6 chars after to gather the things like course number codes
        start_index=potentials_index.end()
        x=parsedText[start_index:start_index+4]# this is just test example, some syllabusses have multiple occurences of CSC so we would need a way to differentiate 


    prereqs=re.finditer("Prerequisites|prerequisites",parsedText)#either need to gather the 3 char class codes and or the entire CSC XYZ 

    for prereqs_index in prereqs:#prereqs indexes
        prereqs_start_index=prereqs_index.end()
        y=parsedText[prereqs_start_index:prereqs_start_index+30]

    return {
    "course_id": x,
    "prerequisite_text": y
}
    

from pathlib import Path

# folder of .txt files made by extract_syllabi_text.py
TEXT_DIR = Path(__file__).resolve().parent.parent / "inputs" / "syllabi_text"

for txt_file in sorted(TEXT_DIR.glob("*.txt")):
    text = txt_file.read_text(encoding="utf-8")
    result = extract(text)
    print(txt_file.name, "->", result)


#BUILDING PROFILE LOOK AT THE OUTCOMES AND WHAT THEY ARE FOR EACH CLASS,. FOR EXAMPLE INTRO CLASS MEANS YOU WONT BE A MASTER BUT YOU WILL JUST BECOME FAMILIAR WITH IT 

