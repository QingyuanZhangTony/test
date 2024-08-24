import base64
import os
from github import Github
import streamlit as st

#from account_credentials import GITHUB_TOKEN, REPO_NAME


GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
REPO_NAME = st.secrets["REPO_NAME"]

# 初始化 GitHub 客户端
g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)


def upload_file_to_github(local_file_path, repo_file_path, commit_message="Upload file to GitHub"):
    """
    上传文件到 GitHub 仓库中。

    Args:
    - local_file_path (str): 本地文件的路径。
    - repo_file_path (str): 仓库中的目标文件路径。
    - commit_message (str): 提交的备注信息。

    Returns:
    - None
    """
    try:
        # 读取本地文件内容
        with open(local_file_path, "rb") as file:
            content = file.read()

        # 确保 repo_file_path 使用正斜杠
        repo_file_path = repo_file_path.replace("\\", "/")

        # 检查文件是否已经存在
        try:
            contents = repo.get_contents(repo_file_path)
            repo.update_file(contents.path, commit_message, content, contents.sha)
        except Exception:
            repo.create_file(repo_file_path, commit_message, content)

        print(f"File '{local_file_path}' uploaded to GitHub as '{repo_file_path}'")

    except Exception as e:
        print(f"An error occurred while uploading the file '{local_file_path}' to '{repo_file_path}': {str(e)}")


def download_file_from_github(repo_file_path, local_file_path):
    """
    从 GitHub 仓库中下载文件到本地。

    Args:
    - repo_file_path (str): 仓库中的文件路径。
    - local_file_path (str): 本地目标文件路径。

    Returns:
    - None
    """
    try:
        # 确保 repo_file_path 使用正斜杠
        repo_file_path = repo_file_path.replace("\\", "/")

        contents = repo.get_contents(repo_file_path)
        file_content = base64.b64decode(contents.content)

        # 将内容写入本地文件
        with open(local_file_path, "wb") as file:
            file.write(file_content)

        print(f"File '{repo_file_path}' downloaded from GitHub to '{local_file_path}'")

    except Exception as e:
        print(f"An error occurred while downloading the file '{repo_file_path}' to '{local_file_path}': {str(e)}")


def read_file_from_github(repo_file_path):
    """
    直接从 GitHub 仓库中读取文件内容。

    Args:
    - repo_file_path (str): 仓库中的文件路径。

    Returns:
    - str: 文件内容作为字符串返回。
    """
    try:
        # 确保 repo_file_path 使用正斜杠
        repo_file_path = repo_file_path.replace("\\", "/")

        contents = repo.get_contents(repo_file_path)
        file_content = base64.b64decode(contents.content).decode("utf-8")
        return file_content

    except Exception as e:
        print(f"An error occurred while reading the file from GitHub at '{repo_file_path}': {str(e)}")
        return None


def write_file_to_github(content, repo_file_path, commit_message="Write file to GitHub"):
    """
    将内容直接写入 GitHub 仓库中的文件。

    Args:
    - content (str): 要写入的内容。
    - repo_file_path (str): 仓库中的目标文件路径。
    - commit_message (str): 提交的备注信息。

    Returns:
    - None
    """
    try:
        # 确保 repo_file_path 使用正斜杠
        repo_file_path = repo_file_path.replace("\\", "/")

        content_bytes = content.encode('utf-8')
        try:
            contents = repo.get_contents(repo_file_path)
            repo.update_file(contents.path, commit_message, content_bytes, contents.sha)
        except Exception as e:
            print(f"File not found in repo. Creating a new file at '{repo_file_path}'. Exception: {str(e)}")
            repo.create_file(repo_file_path, commit_message, content_bytes)

        print(f"Content written to GitHub file '{repo_file_path}'")

    except Exception as e:
        print(f"An error occurred while writing to the GitHub file '{repo_file_path}': {str(e)}")


def check_file_exists_in_github(repo_file_path):
    """
    检查 GitHub 仓库中的文件是否存在。

    Args:
    - repo_file_path (str): 仓库中的文件路径。

    Returns:
    - bool: 如果文件存在返回 True，否则返回 False。
    """
    try:
        # 确保 repo_file_path 使用正斜杠
        repo_file_path = repo_file_path.replace("\\", "/")

        repo.get_contents(repo_file_path)
        return True
    except Exception as e:
        print(f"File '{repo_file_path}' does not exist in the GitHub repository. Exception: {str(e)}")
        return False


def delete_file_from_github(repo_file_path, commit_message="Delete file from GitHub"):
    """
    从 GitHub 仓库中删除文件。

    Args:
    - repo_file_path (str): 仓库中的文件路径。
    - commit_message (str): 提交的备注信息。

    Returns:
    - None
    """
    try:
        # 确保 repo_file_path 使用正斜杠
        repo_file_path = repo_file_path.replace("\\", "/")

        # 检查文件是否存在
        contents = repo.get_contents(repo_file_path)
        repo.delete_file(contents.path, commit_message, contents.sha)

        print(f"File '{repo_file_path}' has been deleted from GitHub.")

    except Exception as e:
        print(f"An error occurred while deleting the file '{repo_file_path}' from GitHub: {str(e)}")
