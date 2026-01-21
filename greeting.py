#!/usr/bin/env python3
import argparse

def hello_world(name="world"):
    print(f"hello {name}")

if __name__ == "__main__":
    # Default behavior is to print "hello world"
    parser = argparse.ArgumentParser(description="Print a greeting")
    parser.add_argument("-n", "--name", default="world", help="Name to greet (default: world)")
    args = parser.parse_args()
    
    if args.name == "world":
        print("hello world")
    else:
        hello_world(args.name)
