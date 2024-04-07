import com.alibaba.fastjson.JSONArray;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import main.Rewriter;
import main.Utils;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.hep.HepMatchOrder;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.externalize.RelWriterImpl;
import org.apache.calcite.rel.rel2sql.RelToSqlConverter;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.sql.dialect.PostgresqlSqlDialect;

import java.io.PrintWriter;
import java.lang.reflect.Type;
import java.util.List;
import java.util.Scanner;

public class get_physical_tree {
    public static void main(String[] args) throws Exception{
        String path = System.getProperty("user.dir");
        String[] levels = path.split("/");
        // Use StringBuilder to rebuild the path without the last level
        StringBuilder modifiedPath = new StringBuilder();
        for(int i = 0; i < levels.length - 1; i++) {
            modifiedPath.append(levels[i]);
            // Add "/" between the levels but not at the end
            if(i < levels.length - 2) {
                modifiedPath.append("/");
            }
        }
        String newpath = modifiedPath.toString();
        Scanner scanner = new Scanner(System.in);
        String inputs = scanner.nextLine();
        Gson gson = new Gson();
        Type type = new TypeToken<List<Object>>(){}.getType();
        List<Object> inputList = gson.fromJson(inputs, type);
        String db_id = (String) inputList.get(0);
        String testSql = (String) inputList.get(1);
        testSql = testSql.replace(";", "");
        JSONArray schemaJson = Utils.readJsonFile(newpath+"/data/data_llmr2/schemas/" + db_id + ".json");
        Rewriter rewriter = new Rewriter(schemaJson);
        RelNode testRelNode = rewriter.SQL2RA(testSql);
        final RelWriter relWriter = new RelWriterImpl(new PrintWriter(System.out), SqlExplainLevel.ALL_ATTRIBUTES, false);
        testRelNode.explain(relWriter);
    }
}
