import logging
import json
import os
from datetime import datetime
from telethon import TelegramClient, events
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



# Configuration
API_ID=''  # Replace with your actual API ID
API_HASH=''  # Replace with your actual API Hash
# API_ID = os.getenv('API_ID')  # Retrieve the API ID
# API_HASH = os.getenv('API_HASH')  # Retrieve the API HASH
SCRAPE_LIMIT = 100  # Number of messages to scrape per channel
CHANNELS = [
    'https://t.me/DoctorsET',
    'https://t.me/CheMed123',  # Verify correct Chemed channel URL
    'https://t.me/lobelia4cosmetics',
    'https://t.me/yetenaweg',
    'https://t.me/EAHCI'
]

# File and directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
MEDIA_DIR = os.path.join(RAW_DATA_DIR, 'media')
LOG_FILE = os.path.join(BASE_DIR, 'scraping.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

async def scrape_channel(client, channel_link):
    """Scrape messages and media from a Telegram channel"""
    channel_name = channel_link.split('/')[-1]
    logging.info(f"Starting scrape for channel: {channel_name}")
    
    try:
        # Create directory structure
        channel_media_dir = os.path.join(MEDIA_DIR, channel_name)
        os.makedirs(channel_media_dir, exist_ok=True)
        
        # Get messages
        messages = await client.get_messages(channel_link, limit=SCRAPE_LIMIT)
        scraped_data = []
        
        for message in messages:
            if not message:  # Skip empty messages
                continue
                
            # Base message data
            message_data = {
                'channel': channel_name,
                'message_id': message.id,
                'date': message.date.isoformat() if message.date else None,
                'text': message.text,
                'media_path': None,
                'scrape_timestamp': datetime.utcnow().isoformat()
            }
            
            # Handle media
            if message.media:
                try:
                    file_ext = message.file.ext if message.file else '.bin'
                    media_filename = f"{message.id}_{message.date.strftime('%Y%m%d')}{file_ext}"
                    media_path = os.path.join(channel_media_dir, media_filename)
                    
                    await message.download_media(file=media_path)
                    message_data['media_path'] = media_path
                    logging.debug(f"Downloaded media: {media_filename}")
                except Exception as media_error:
                    logging.error(f"Media download failed for message {message.id}: {media_error}")
            
            scraped_data.append(message_data)
        
        # Save metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RAW_DATA_DIR, f"{channel_name}_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scraped_data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Successfully scraped {len(scraped_data)} messages from {channel_name}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to scrape {channel_name}: {str(e)}")
        return False

async def main():
    # Create data directories
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(MEDIA_DIR, exist_ok=True)
    
    # Initialize Telegram client
    client = TelegramClient('kara_solutions_session', API_ID, API_HASH)
    
    try:
        await client.start()
        logging.info("Telegram client started successfully")
        
        # Scrape all channels
        for channel in CHANNELS:
            await scrape_channel(client, channel)
            
    except Exception as e:
        logging.error(f"Fatal error in main process: {str(e)}")
    finally:
        await client.disconnect()
        logging.info("Telegram client disconnected")

if __name__ == '__main__':
    # Run the scraper
    import asyncio
    asyncio.run(main())