from supabase import create_client
import os

def get_supabase():
    return create_client(
        os.getenv("https://wujohognxfjpaxnupfuv.supabase.co"),
        os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind1am9ob2dueGZqcGF4bnVwZnV2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE3MTc4NTAsImV4cCI6MjA4NzI5Mzg1MH0.9zjizDQjQLp3V4ggfIvJ0m0AGF4muUISUDkrr6FhPxc")
    )
