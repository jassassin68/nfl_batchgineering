# test_env_loading.py
from pathlib import Path
from dotenv import load_dotenv
import os
import snowflake.connector



# Test the same logic as the script
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'

print(f"Looking for .env at: {env_path}")
print(f".env exists: {env_path.exists()}")

if env_path.exists():
    load_dotenv(env_path)
    print("✅ .env loaded successfully")
    
    # Check key variables
    account = os.getenv('SNOWFLAKE_ACCOUNT')
    user = os.getenv('SNOWFLAKE_USER')
    
    print(f"SNOWFLAKE_ACCOUNT: {'✅ Found' if account else '❌ Missing'}")
    print(f"SNOWFLAKE_USER: {'✅ Found' if user else '❌ Missing'}")
else:
    print("❌ .env file not found")

def test_connection_formats():
    """Test different account format variations"""
    
    base_account = os.getenv('SNOWFLAKE_ACCOUNT')
    username = os.getenv('SNOWFLAKE_USER')
    password = os.getenv('SNOWFLAKE_PASSWORD')
    
    print(f"🔍 Original account from .env: {base_account}")
    print(f"👤 Username: {username}")
    print(f"🔑 Password: {'***' if password else 'MISSING'}")
    
    # Different account formats to try
    account_formats = []
    
    if base_account:
        # If it contains dots, try variations
        if '.' in base_account:
            parts = base_account.split('.')
            account_formats.extend([
                base_account,  # Full format as-is
                parts[0],      # Just account ID
                f"{parts[0]}.{parts[1]}" if len(parts) > 1 else parts[0]  # Account + region
            ])
        else:
            account_formats.append(base_account)
            
        # Add common variations
        if not base_account.endswith('.snowflakecomputing.com'):
            account_formats.append(f"{base_account}.snowflakecomputing.com")
    
    # Remove duplicates while preserving order
    account_formats = list(dict.fromkeys(account_formats))
    
    print(f"\n🧪 Testing {len(account_formats)} account format(s):")
    
    for i, account in enumerate(account_formats, 1):
        print(f"\n--- Test {i}/{len(account_formats)} ---")
        print(f"🔌 Testing account format: {account}")
        
        try:
            conn = snowflake.connector.connect(
                account=account,
                user=username,
                password=password,
                role='ACCOUNTADMIN',
                login_timeout=30,
                network_timeout=30,
                connect_timeout=30
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT CURRENT_USER(), CURRENT_ACCOUNT(), CURRENT_REGION()")
            user, account_name, region = cursor.fetchone()
            
            print(f"✅ SUCCESS!")
            print(f"   Connected User: {user}")
            print(f"   Account Name: {account_name}")
            print(f"   Region: {region}")
            print(f"   🎯 USE THIS FORMAT: {account}")
            
            conn.close()
            return account  # Return successful format
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ FAILED: {error_msg[:100]}...")
            
            # Provide specific troubleshooting based on error
            if "250001" in error_msg:
                print("   💡 This is a network connectivity error")
                print("   💡 Check firewall/VPN settings")
            elif "authentication" in error_msg.lower():
                print("   💡 This is an authentication error")
                print("   💡 Verify username/password")
            elif "account" in error_msg.lower():
                print("   💡 This is an account identifier error")
                print("   💡 Try a different account format")
    
    print(f"\n❌ All connection attempts failed!")
    return None

def test_network_connectivity():
    """Test basic network connectivity to Snowflake"""
    import socket
    
    print(f"\n🌐 Testing network connectivity...")
    
    account = os.getenv('SNOWFLAKE_ACCOUNT')
    if not account:
        print("❌ No account specified")
        return
    
    # Extract hostname
    if not account.endswith('.snowflakecomputing.com'):
        hostname = f"{account}.snowflakecomputing.com"
    else:
        hostname = account
    
    print(f"🔍 Testing connection to: {hostname}")
    
    try:
        # Test DNS resolution
        ip = socket.gethostbyname(hostname)
        print(f"✅ DNS resolution successful: {hostname} -> {ip}")
        
        # Test port connectivity
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((hostname, 443))
        sock.close()
        
        if result == 0:
            print(f"✅ Port 443 (HTTPS) is reachable")
        else:
            print(f"❌ Port 443 (HTTPS) is NOT reachable")
            print(f"💡 You may be behind a firewall/proxy")
            
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed: {e}")
        print(f"💡 Check your account identifier format")
    except Exception as e:
        print(f"❌ Network test failed: {e}")

def main():
    print("🏈 Snowflake Connection Debug Tool")
    print("=" * 50)
    
    # Test network connectivity first
    test_network_connectivity()
    
    # Test different connection formats
    successful_format = test_connection_formats()
    
    if successful_format:
        print(f"\n🎉 SUCCESS! Use this account format in your .env file:")
        print(f"SNOWFLAKE_ACCOUNT={successful_format}")
    else:
        print(f"\n❌ No connection format worked. Check:")
        print(f"   1. Are you behind a corporate firewall?")
        print(f"   2. Can you access Snowflake web UI?")
        print(f"   3. Are your credentials correct?")
        print(f"   4. Try from a different network (e.g., mobile hotspot)")

if __name__ == "__main__":
    main()