import asyncio
from playwright.async_api import async_playwright

async def generate_pdf():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("file:///d:/Fertilizer%20detection/AgriAI_Technical_Report.html", wait_until="networkidle")
        await page.pdf(
            path="d:/Fertilizer detection/AgriAI_Technical_Report.pdf",
            format="A4",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}
        )
        await browser.close()
        print("PDF generated successfully.")

asyncio.run(generate_pdf())
