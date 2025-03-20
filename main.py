import time

from text_processor import text_processor_run
from image_processor import image_processor_run
from blob_processor import blob_processor_run
from answer_generator_new import AnswerGenerator
from report_generator_new import ReportGenerator


class Dossier:
    def __init__(self):
        pass

    def text_processor(self):
        text_processor_run()

    def image_processor(self):
        image_processor_run()

    def blob_processor(self):
        blob_processor_run()

    def answer_generator(self):
        # project_name = "nox_medical_nox_a1_and_t3"
        project_name = "hologic_genius_ai_detection"
        ans_generator = AnswerGenerator(project_name)
        ans_generator.answer_generator_run()

    def report_generator(self):
        report_generator = ReportGenerator()
        report_generator.generate_report_run()

    def run_workflow(self):
        print("Starting workflow...")
        start_time = time.time()

        print("Step 1: Running image processor...")
        self.image_processor()
        print("Image processing complete. Moving to text processing.\n")

        print("Step 2: Running text processor...\n")
        self.text_processor()
        print("Text processing complete. Moving to blob processing.\n")

        print("Step 3: Running blob processor...")
        self.blob_processor()
        print("Blob processing complete. Moving to answer generation.\n")

        print("Step 4: Running answer generator...")
        self.answer_generator()
        print("Answer generation complete. Moving to report generation.\n")

        print("Step 5: Running report generator...")
        self.report_generator()
        print("Report generation complete. Workflow finished!\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            f"Total execution time: {int(hours)} hrs {int(minutes)} mins {seconds:.2f} secs")

if __name__ == "__main__":
    app = Dossier()
    app.run_workflow()
