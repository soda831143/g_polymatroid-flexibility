# imports

try:
    from . import tools
except ImportError as e:
    print(f"警告: 无法导入tools模块 (Gurobi可能未安装): {e}")
    
from . import algo_template
from . import algo_no_flex
from . import algo_exact
from . import algo_Barot_wo_pc
try:
    from . import algo_Barot_w_pc_KR
except ImportError:
    pass
try:
    from . import algo_Barot_w_pc
except ImportError:
    pass
try:
    from . import algo_Homothet_Projection
except ImportError:
    pass
from . import algo_Inner_Homothets
try:
    from . import algo_Outer_Homothets
except ImportError:
    pass
try:
    from . import algo_Union_Homothets_Stage_0
except ImportError:
    pass
try:
    from . import algo_Union_Homothets_Stage_1
except ImportError:
    pass
try:
    from . import algo_Zhen_Inner
except ImportError:
    pass
from . import algo_Zonotope_l1
from . import algo_Zonotope_l2
from . import algo_Zonotope_Rel
from . import algo_Zonotope
from . import algo_Barot_Inner
from . import algo_Inner_affine

# G-Polymatroid坐标变换算法 (统一框架)
from . import algo_g_polymatroid_transform_det  # 确定性版本 (正确物理模型)
from . import algo_g_polymatroid_jcc_sro        # SRO版本
from . import algo_g_polymatroid_jcc_resro      # Re-SRO版本
