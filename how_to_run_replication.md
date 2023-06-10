# How to Run Replication Notebooks

Note: please send a calendar invite to stefan.j.wojcik@gmail.com so he can power on the VM before you begin running the following steps. 

- Using VSCODE,  SSH into the VM (ensure IP address is updated in ~/.ssh/config). The IP address currently is: 
- Once in the VM, open a terminal, ensure the Bash terminal says `nikolas@[ipaddress]`
- Type `cd /home/ubuntu/facing_voters` into the Bash terminal.
- Now type `julia --project=.` and press enter
- You should now have a Julia REPL in front of you. 
- In the Julia REPL, type `using Pluto; Pluto.run()`
- A web page should automatically open to Pluto. You should see links in that page to [notebook1.jl]() and [notebook2.jl]()
- Please wait for the notebooks page to load. This may take a few minutes.
- There will be a number of recent notebooks listed, but you should manually enter the paths to each of the notebooks. The correct path are listed below:
    - Path to Notebook 1: /home/ubuntu/facing_voters/src/notebooks/notebook1.jl
    - Path to Notebook 2: /home/ubuntu/facing_voters/src/notebooks/notebook2.jl
- You will need to enter the path to each notebook one at a time. Notebook 1 contains mainly R-based results, and should complete in 30 minutes or less. Notebook 2 contains mainly Julia-based results, and should complete in 1 hour or less. If it takes longer than this, it's not necessarily a problem, but please let us know.
