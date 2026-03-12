# run_app.py
import os
import sys
import traceback

def main():
    try:
        from streamlit.web import cli as stcli
        
        # 切换到脚本所在目录，确保相对路径正确
        base_path = os.path.dirname(__file__)
        os.chdir(base_path)
        print(f"工作目录已切换到: {base_path}", file=sys.stderr)
        
        # 构建启动参数（去掉 headless，让浏览器自动打开）
        sys.argv = ["streamlit", "run", "app.py"]
        
        # 启动 Streamlit
        sys.exit(stcli.main())
    except Exception as e:
        print("\n❌ 启动 Streamlit 应用时出错：", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("\n按回车键退出...", file=sys.stderr)
        input()  # 等待用户输入，以便查看错误
        sys.exit(1)

if __name__ == '__main__':
    main()