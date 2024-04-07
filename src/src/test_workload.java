import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import main.Node;
import main.Rewriter;
import main.Utils;
import org.apache.calcite.rel.RelNode;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.lang.StringEscapeUtils;



public class test_workload {
    public static void main(String[] args) throws Exception {

        //DB Config
        JSONArray schemaJson = Utils.readJsonFile(args[1]);
        Rewriter rewriter = new Rewriter(schemaJson);
        String[] workload = Utils.readWorkloadFromFile(args[0]);
        System.out.println(workload.length);
        List addedWorkload = new ArrayList();
        JSONArray rewrittenList = new JSONArray();
        List unRewrittenList = new ArrayList();
        List failureList = new ArrayList();


        for (int i = 0; i < workload.length; i++) {
            String sql = workload[i];
            if (addedWorkload.contains(sql)) {
                continue;
            }
            System.out.println("\u001B[1;31m" + "-------------------------------------------正在改写："+ i + "\u001B[0m");
            try {
                RelNode originRelNode = rewriter.SQL2RA(sql);
                double origin_cost = rewriter.getCostRecordFromRelNode(originRelNode);
                Node resultNode = new Node(sql, originRelNode, (float) origin_cost, rewriter, (float) 0.1,null,"original query");
                Node res = resultNode.UTCSEARCH(20, resultNode,1);
                String rewritten_sql = res.state;
                if (!rewritten_sql.equalsIgnoreCase(sql)) {
                    JSONObject dataJson = new JSONObject();
                    dataJson.put("origin_cost", String.format("%.4f",origin_cost));
                    dataJson.put("origin_sql", sql);
                    dataJson.put("rewritten_cost", String.format("%.4f",rewriter.getCostRecordFromRelNode(res.state_rel)));
                    dataJson.put("rewritten_sql", res.state);
                    rewrittenList.add(dataJson);
                }else {
                    unRewrittenList.add(sql);
                }
            } catch (Exception error) {
                System.out.println(error.toString());
                failureList.add(sql);
            }
            addedWorkload.add(sql);
        }
        JSONObject resultJson = new JSONObject();
        resultJson.put("rewritten", rewrittenList);
        resultJson.put("unRewritten", unRewrittenList);
        resultJson.put("failure", failureList);
        Utils.writeContentStringToLocalFile(resultJson.toJSONString(), args[2]);
    }
}
